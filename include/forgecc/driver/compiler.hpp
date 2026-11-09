#pragma once

#include "forgecc/codegen/codegen_options.hpp"
#include "forgecc/native/native_compiler.hpp"
#include "forgecc/native/native_executor.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace forgecc::driver {

struct SourceInput {
    std::filesystem::path path;
    std::string contents;
};

struct CompilationResult {
    bool success = false;
    std::filesystem::path executablePath;
    std::string diagnostics;
    std::string output;
    int exitCode = -1;
};

class CompilerDriver {
public:
    explicit CompilerDriver(SourceInput source) : source_(std::move(source)) {}

    [[nodiscard]] const SourceInput& source() const {
        return source_;
    }

    [[nodiscard]] native::NativeCompilationResult compileGeneratedSource(
        std::string_view generatedSource, const std::filesystem::path& outputPath,
        native::NativeCompilerOptions options = {}) const;

    [[nodiscard]] CompilationResult compileAndRun(
        const std::filesystem::path& outputPath,
        native::NativeCompilerOptions nativeOptions = {},
        codegen::CodegenOptions codegenOptions = {}) const;

private:
    SourceInput source_;
};

} // namespace forgecc::driver
