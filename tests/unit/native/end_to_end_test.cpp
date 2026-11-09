#include "forgecc/driver/compiler.hpp"

#include <cassert>
#include <filesystem>
#include <string>

namespace {

std::filesystem::path outputPath(const char* name) {
    return std::filesystem::current_path() / name;
}

void removeOutput(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void testReluRunsThroughTheCompletePipeline() {
    const auto output = outputPath("forgecc-e2e-relu");
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({
        "relu.fcc",
        "A: f32[2, 2]\nout = relu(A)\n",
    });

    const auto result = compiler.compileAndRun(output);
    assert(result.success);
    assert(result.exitCode == 0);
    assert(result.output == "0\n2\n0\n4\n");
    assert(std::filesystem::is_regular_file(result.executablePath));
    removeOutput(output);
}

void testMatmulRunsThroughTheCompletePipeline() {
    const auto output = outputPath("forgecc-e2e-matmul");
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({
        "matmul.fcc",
        "A: f32[2, 2]\nB: f32[2, 2]\nout = matmul(A, B)\n",
    });

    const auto result = compiler.compileAndRun(output);
    assert(result.success);
    assert(result.output == "-5\n6\n-9\n10\n");
    removeOutput(output);
}

void testPipelineStopsOnInvalidSource() {
    const auto output = outputPath("forgecc-e2e-invalid");
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({"invalid.fcc", "A: f32[2, 2]\nout = nope(A)\n"});

    const auto result = compiler.compileAndRun(output);
    assert(!result.success);
    assert(result.diagnostics.find("semantic analysis failed") != std::string::npos);
    assert(!std::filesystem::exists(output));
}

void testCombinedMatmulAddAndReluRunsCorrectly() {
    const auto output = outputPath("forgecc-e2e-combined");
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({
        "combined.fcc",
        "A: f32[2, 2]\nB: f32[2, 2]\nbias: f32[2, 2]\n"
        "out = relu(matmul(A, B) + bias)\n",
    });

    const auto result = compiler.compileAndRun(output);
    assert(result.success);
    assert(result.output == "0\n8\n0\n14\n");
    removeOutput(output);
}

} // namespace

void runEndToEndTests() {
    testReluRunsThroughTheCompletePipeline();
    testMatmulRunsThroughTheCompletePipeline();
    testPipelineStopsOnInvalidSource();
    testCombinedMatmulAddAndReluRunsCorrectly();
}
