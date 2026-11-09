#include "forgecc/native/native_executor.hpp"

#include <cerrno>
#include <cstring>
#include <array>
#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>

namespace forgecc::native {

namespace {

std::string systemError(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

std::string readPipe(int descriptor) {
    std::string output;
    std::array<char, 4096> buffer{};
    while (true) {
        const ssize_t count = read(descriptor, buffer.data(), buffer.size());
        if (count > 0) {
            output.append(buffer.data(), static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
    return output;
}

} // namespace

NativeExecutionResult NativeExecutor::execute(
    const std::filesystem::path& executablePath) const {
    if (executablePath.empty()) {
        return {false, -1, "native executable path must not be empty"};
    }

    std::error_code filesystemError;
    if (!std::filesystem::is_regular_file(executablePath, filesystemError) ||
        filesystemError) {
        return {false, -1, "native executable does not exist: " + executablePath.string()};
    }
    if (access(executablePath.c_str(), X_OK) != 0) {
        return {false, -1, systemError("native executable is not executable")};
    }

    const std::filesystem::path absolutePath = std::filesystem::absolute(executablePath,
                                                                           filesystemError);
    if (filesystemError) {
        return {false, -1, "could not resolve native executable path: " +
                              filesystemError.message()};
    }

    int outputPipe[2] = {-1, -1};
    if (pipe(outputPipe) != 0) {
        return {false, -1, systemError("could not create native execution pipe")};
    }

    const pid_t child = fork();
    if (child < 0) {
        close(outputPipe[0]);
        close(outputPipe[1]);
        return {false, -1, systemError("could not start native executable")};
    }

    if (child == 0) {
        close(outputPipe[0]);
        if (dup2(outputPipe[1], STDOUT_FILENO) < 0 ||
            dup2(outputPipe[1], STDERR_FILENO) < 0) {
            const std::string message =
                systemError("could not redirect native executable output") + "\n";
            write(STDERR_FILENO, message.data(), message.size());
            _exit(126);
        }
        close(outputPipe[1]);
        execl(absolutePath.c_str(), absolutePath.c_str(), nullptr);
        const std::string message = systemError("could not execute native executable") + "\n";
        write(STDERR_FILENO, message.data(), message.size());
        _exit(127);
    }

    close(outputPipe[1]);
    const std::string output = readPipe(outputPipe[0]);
    close(outputPipe[0]);

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            return {false, -1, systemError("could not wait for native executable")};
        }
    }

    if (!WIFEXITED(status)) {
        return {false, -1, output + "native executable terminated abnormally\n"};
    }

    const int exitCode = WEXITSTATUS(status);
    return {exitCode == 0, exitCode, output};
}

} // namespace forgecc::native
