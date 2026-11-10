#include "correctness_reference.hpp"

#include "forgecc/driver/compiler.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using forgecc::test::TensorReference;

std::filesystem::path outputPath(const std::string& name) {
    return std::filesystem::current_path() / ("forgecc-correctness-" + name);
}

void removeOutput(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

std::vector<float> runProgram(const std::string& name, const std::string& source,
                              const std::vector<float>& expected) {
    const auto output = outputPath(name);
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({name + ".fcc", source});
    const auto result = compiler.compileAndRun(output);
    if (!result.success) {
        std::cerr << result.diagnostics;
    }
    assert(result.success);
    assert(result.exitCode == 0);

    const auto actual = forgecc::test::parseOutput(result.output);
    assert(actual.has_value());
    const auto comparison = forgecc::test::compare(expected, *actual);
    if (!comparison.passed) {
        std::cerr << comparison.failure << '\n';
    }
    assert(comparison.passed);
    assert(comparison.comparedElements == expected.size());
    removeOutput(output);
    return *actual;
}

void testElementwiseOperations() {
    const TensorReference left = forgecc::test::deterministicTensor({2, 3});
    const TensorReference right = forgecc::test::deterministicTensor({2, 3});
    for (const auto [operation, symbol] : std::vector<std::pair<std::string, char>>{
             {"add", '+'}, {"subtract", '-'}, {"multiply", '*'}, {"divide", '/'}}) {
        const std::string source = std::string("A: f32[2, 3]\nB: f32[2, 3]\nout = A ") + symbol +
                                   " B\n";
        const auto expected = forgecc::test::elementwise(left, right, symbol);
        runProgram(operation, source, expected.data);
    }
}

void testReluWithNegativePositiveAndZeroValues() {
    const TensorReference input = forgecc::test::deterministicTensor({2, 3});
    const auto expected = forgecc::test::relu(input);
    runProgram("relu", "A: f32[2, 3]\nout = relu(A)\n", expected.data);
}

void testMatrixMultiplicationAtMultipleSizes() {
    for (const auto [rows, shared, columns] :
         std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>{
             {2, 2, 2}, {2, 3, 2}, {3, 2, 4}, {1, 3, 1}, {4, 3, 5}}) {
        const TensorReference left = forgecc::test::deterministicTensor({rows, shared});
        const TensorReference right = forgecc::test::deterministicTensor({shared, columns});
        const auto expected = forgecc::test::matmul(left, right);
        const std::string name = "matmul-" + std::to_string(rows) + "-" +
                                 std::to_string(shared) + "-" + std::to_string(columns);
        const std::string source = "A: f32[" + std::to_string(rows) + ", " +
                                   std::to_string(shared) + "]\nB: f32[" +
                                   std::to_string(shared) + ", " + std::to_string(columns) +
                                   "]\nout = matmul(A, B)\n";
        runProgram(name, source, expected.data);
    }
}

void testCombinedComputation() {
    const TensorReference left = forgecc::test::deterministicTensor({2, 2});
    const TensorReference right = forgecc::test::deterministicTensor({2, 2});
    const TensorReference bias = forgecc::test::deterministicTensor({2, 2});
    const auto product = forgecc::test::matmul(left, right);
    const auto shifted = forgecc::test::elementwise(product, bias, '+');
    const auto expected = forgecc::test::relu(shifted);
    runProgram("combined", "A: f32[2, 2]\nB: f32[2, 2]\nbias: f32[2, 2]\n"
                          "out = relu(matmul(A, B) + bias)\n",
               expected.data);
}

void testInvalidProgramsFailBeforeExecution() {
    const auto output = outputPath("invalid");
    removeOutput(output);
    forgecc::driver::CompilerDriver compiler({
        "invalid.fcc", "A: f32[2, 3]\nB: f32[4, 2]\nout = matmul(A, B)\n"});
    const auto result = compiler.compileAndRun(output);
    assert(!result.success);
    assert(result.diagnostics.find("semantic analysis failed") != std::string::npos);
    assert(!std::filesystem::exists(output));
}

void testNumericalUtilitiesRejectMalformedOrWrongOutput() {
    assert(!forgecc::test::parseOutput("1.0\ninvalid\n").has_value());
    const auto comparison = forgecc::test::compare({1.0F, 2.0F}, {1.0F, 2.1F});
    assert(!comparison.passed);
    assert(comparison.comparedElements == 2);
    assert(comparison.mismatchedElements == 1);
    assert(comparison.maximumAbsoluteError > 0.09F);
}

} // namespace

void runCorrectnessTests() {
    testElementwiseOperations();
    testReluWithNegativePositiveAndZeroValues();
    testMatrixMultiplicationAtMultipleSizes();
    testCombinedComputation();
    testInvalidProgramsFailBeforeExecution();
    testNumericalUtilitiesRejectMalformedOrWrongOutput();
}
