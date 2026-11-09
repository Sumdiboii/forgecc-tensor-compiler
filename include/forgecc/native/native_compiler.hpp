#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace forgecc::native {

struct NativeCompilerOptions {
    std::string compiler = "g++";
};

struct NativeCompilationResult {
    bool success = false;
    std::filesystem::path outputPath;
    std::string diagnostics;
    int exitCode = -1;
};

class NativeCompiler {
public:
    explicit NativeCompiler(NativeCompilerOptions options = {})
        : options_(std::move(options)) {}

    [[nodiscard]] NativeCompilationResult compile(
        std::string_view generatedSource, const std::filesystem::path& outputPath) const;

private:
    NativeCompilerOptions options_;
};

} // namespace forgecc::native
