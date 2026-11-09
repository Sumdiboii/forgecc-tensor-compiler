#include "forgecc/driver/compiler.hpp"

namespace forgecc::driver {

native::NativeCompilationResult CompilerDriver::compileGeneratedSource(
    std::string_view generatedSource, const std::filesystem::path& outputPath,
    native::NativeCompilerOptions options) const {
    return native::NativeCompiler(std::move(options)).compile(generatedSource, outputPath);
}

} // namespace forgecc::driver
