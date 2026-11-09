#include "forgecc/native/native_compiler.hpp"

#include <cerrno>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <utility>
#include <vector>

namespace forgecc::native {

namespace {

struct TemporarySourceFile {
    std::filesystem::path path;
    int descriptor = -1;

    ~TemporarySourceFile() {
        if (descriptor >= 0) {
            close(descriptor);
        }
        if (!path.empty()) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

NativeCompilationResult failure(const std::filesystem::path& outputPath,
                                std::string message) {
    return {false, outputPath, std::move(message), -1};
}

std::string systemError(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

bool writeAll(int descriptor, std::string_view source, std::string& error) {
    std::size_t written = 0;
    while (written < source.size()) {
        const ssize_t result = write(descriptor, source.data() + written, source.size() - written);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = systemError("could not write generated source");
            return false;
        }
        if (result == 0) {
            error = "could not write generated source: write returned zero";
            return false;
        }
        written += static_cast<std::size_t>(result);
    }
    return true;
}

bool createTemporarySource(std::string_view source, TemporarySourceFile& temporary,
                           std::string& error) {
    std::error_code directoryError;
    const auto directory = std::filesystem::temp_directory_path(directoryError);
    if (directoryError) {
        error = "could not determine temporary directory: " + directoryError.message();
        return false;
    }

    std::string pattern = (directory / "forgecc-generated-XXXXXX.cpp").string();
    std::vector<char> mutablePattern(pattern.begin(), pattern.end());
    mutablePattern.push_back('\0');

    temporary.descriptor = mkstemps(mutablePattern.data(), 4);
    if (temporary.descriptor < 0) {
        error = systemError("could not create temporary source file");
        return false;
    }
    temporary.path = mutablePattern.data();

    if (!writeAll(temporary.descriptor, source, error)) {
        return false;
    }
    if (close(temporary.descriptor) != 0) {
        temporary.descriptor = -1;
        error = systemError("could not close generated source file");
        return false;
    }
    temporary.descriptor = -1;
    return true;
}

std::string readDiagnostics(int descriptor) {
    std::string diagnostics;
    std::array<char, 4096> buffer{};
    while (true) {
        const ssize_t count = read(descriptor, buffer.data(), buffer.size());
        if (count > 0) {
            diagnostics.append(buffer.data(), static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
    return diagnostics;
}

} // namespace

NativeCompilationResult NativeCompiler::compile(
    std::string_view generatedSource, const std::filesystem::path& outputPath) const {
    if (options_.compiler.empty()) {
        return failure(outputPath, "native compiler path must not be empty");
    }
    if (outputPath.empty()) {
        return failure(outputPath, "native output path must not be empty");
    }
    std::error_code filesystemError;
    const bool outputExists = std::filesystem::exists(outputPath, filesystemError);
    if (filesystemError) {
        return failure(outputPath,
                       "could not inspect native output path: " + filesystemError.message());
    }
    if (outputExists) {
        return failure(outputPath, "native output path already exists: " + outputPath.string());
    }

    const auto parent = outputPath.parent_path();
    if (!parent.empty()) {
        filesystemError.clear();
        const bool parentIsDirectory = std::filesystem::is_directory(parent, filesystemError);
        if (filesystemError || !parentIsDirectory) {
            return failure(outputPath, "native output directory does not exist: " + parent.string());
        }
    }

    TemporarySourceFile temporary;
    std::string error;
    if (!createTemporarySource(generatedSource, temporary, error)) {
        return failure(outputPath, std::move(error));
    }

    int diagnosticsPipe[2] = {-1, -1};
    if (pipe(diagnosticsPipe) != 0) {
        return failure(outputPath, systemError("could not create compiler diagnostics pipe"));
    }

    const pid_t child = fork();
    if (child < 0) {
        close(diagnosticsPipe[0]);
        close(diagnosticsPipe[1]);
        return failure(outputPath, systemError("could not start native compiler"));
    }

    if (child == 0) {
        close(diagnosticsPipe[0]);
        if (dup2(diagnosticsPipe[1], STDOUT_FILENO) < 0 ||
            dup2(diagnosticsPipe[1], STDERR_FILENO) < 0) {
            const std::string message = systemError("could not redirect compiler diagnostics") + "\n";
            write(STDERR_FILENO, message.data(), message.size());
            _exit(126);
        }
        close(diagnosticsPipe[1]);

        execlp(options_.compiler.c_str(), options_.compiler.c_str(), "-std=c++20", "-x", "c++",
               temporary.path.c_str(), "-o", outputPath.c_str(), nullptr);
        const std::string message = systemError("could not execute native compiler") + "\n";
        write(STDERR_FILENO, message.data(), message.size());
        _exit(127);
    }

    close(diagnosticsPipe[1]);
    const std::string diagnostics = readDiagnostics(diagnosticsPipe[0]);
    close(diagnosticsPipe[0]);

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            return failure(outputPath, systemError("could not wait for native compiler"));
        }
    }

    if (!WIFEXITED(status)) {
        return failure(outputPath, diagnostics.empty()
                                     ? "native compiler terminated abnormally"
                                     : diagnostics);
    }

    const int exitCode = WEXITSTATUS(status);
    if (exitCode != 0) {
        return {false, outputPath, diagnostics, exitCode};
    }
    filesystemError.clear();
    if (!std::filesystem::is_regular_file(outputPath, filesystemError) || filesystemError) {
        return {false, outputPath,
                "native compiler reported success but did not produce the output executable", 0};
    }

    return {true, outputPath, diagnostics, exitCode};
}

} // namespace forgecc::native
