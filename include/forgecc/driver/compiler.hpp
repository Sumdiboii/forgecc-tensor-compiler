#pragma once

#include "forgecc/native/native_compiler.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace forgecc::driver {

struct SourceInput {
    std::filesystem::path path;
    std::string contents;
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

private:
    SourceInput source_;
};

} // namespace forgecc::driver
