#include "forgecc/driver/compiler.hpp"
#include "forgecc/native/native_compiler.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path testDirectory() {
    const auto directory = std::filesystem::temp_directory_path() / "forgecc native tests";
    std::filesystem::create_directories(directory);
    return directory;
}

void removeOutput(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void testCompilesValidSourceAndHandlesSpacesInOutputPath() {
    const auto output = testDirectory() / "valid output";
    removeOutput(output);

    forgecc::native::NativeCompiler compiler;
    const auto result = compiler.compile("int main() { return 0; }\n", output);

    assert(result.success);
    assert(result.exitCode == 0);
    assert(result.outputPath == output);
    assert(std::filesystem::is_regular_file(output));
    removeOutput(output);
}

void testReportsCompilerDiagnostics() {
    const auto output = testDirectory() / "invalid output";
    removeOutput(output);

    forgecc::native::NativeCompiler compiler;
    const auto result = compiler.compile("int main( { return 0; }\n", output);

    assert(!result.success);
    assert(result.exitCode != 0);
    assert(!result.diagnostics.empty());
    assert(!std::filesystem::exists(output));
}

void testReportsMissingCompiler() {
    const auto output = testDirectory() / "missing compiler output";
    removeOutput(output);

    forgecc::native::NativeCompiler compiler({"forgecc-compiler-that-does-not-exist"});
    const auto result = compiler.compile("int main() { return 0; }\n", output);

    assert(!result.success);
    assert(result.exitCode == 127);
    assert(result.diagnostics.find("could not execute native compiler") != std::string::npos);
    assert(!std::filesystem::exists(output));
}

void testRejectsInvalidOutputPath() {
    const auto output = testDirectory() / "missing directory" / "program";
    forgecc::native::NativeCompiler compiler;
    const auto result = compiler.compile("int main() { return 0; }\n", output);

    assert(!result.success);
    assert(result.exitCode == -1);
    assert(result.diagnostics.find("output directory does not exist") != std::string::npos);
}

void testDoesNotOverwriteExistingOutput() {
    const auto output = testDirectory() / "existing output";
    {
        std::ofstream existing(output, std::ios::binary);
        existing << "original";
    }

    forgecc::native::NativeCompiler compiler;
    const auto result = compiler.compile("int main() { return 0; }\n", output);

    assert(!result.success);
    assert(result.diagnostics.find("already exists") != std::string::npos);
    std::ifstream existing(output, std::ios::binary);
    assert(std::string(std::istreambuf_iterator<char>(existing), {}) == "original");
    removeOutput(output);
}

void testDriverForwardsGeneratedSourceWithoutExecutingIt() {
    const auto output = testDirectory() / "driver output";
    removeOutput(output);

    forgecc::driver::CompilerDriver driver({"input.fcc", ""});
    const auto result = driver.compileGeneratedSource("int main() { return 0; }\n", output);

    assert(result.success);
    assert(std::filesystem::is_regular_file(output));
    removeOutput(output);
}

} // namespace

void runNativeCompilerTests() {
    testCompilesValidSourceAndHandlesSpacesInOutputPath();
    testReportsCompilerDiagnostics();
    testReportsMissingCompiler();
    testRejectsInvalidOutputPath();
    testDoesNotOverwriteExistingOutput();
    testDriverForwardsGeneratedSourceWithoutExecutingIt();
}
