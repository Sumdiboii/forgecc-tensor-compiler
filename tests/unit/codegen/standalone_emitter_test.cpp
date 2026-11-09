#include "forgecc/codegen/cpp_emitter.hpp"
#include "forgecc/codegen/standalone_emitter.hpp"
#include "forgecc/ir/graph.hpp"
#include "forgecc/lowering/loop_lowerer.hpp"
#include "forgecc/native/native_compiler.hpp"

#include <cassert>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

forgecc::semantic::TensorType f32(std::vector<std::size_t> dimensions) {
    return {forgecc::semantic::ElementType::F32, std::move(dimensions)};
}

forgecc::lowering::LoopProgram lowerRelu() {
    forgecc::ir::Graph graph;
    const auto input = graph.addInput("A", f32({2, 2}));
    const auto node = graph.addNode(forgecc::ir::OperationKind::Relu, {input}, f32({2, 2}));
    graph.setValueName(graph.node(node).output, "C");
    graph.addOutput(graph.node(node).output);

    forgecc::lowering::LoopLowerer lowerer;
    auto program = lowerer.lower(graph);
    assert(program.has_value());
    return std::move(*program);
}

forgecc::lowering::LoopProgram lowerMatmulRelu() {
    forgecc::ir::Graph graph;
    const auto left = graph.addInput("A", f32({2, 3}));
    const auto right = graph.addInput("B", f32({3, 2}));
    const auto matmul = graph.addNode(forgecc::ir::OperationKind::Matmul, {left, right},
                                      f32({2, 2}));
    graph.setValueName(graph.node(matmul).output, "M");
    const auto relu = graph.addNode(forgecc::ir::OperationKind::Relu,
                                    {graph.node(matmul).output}, f32({2, 2}));
    graph.setValueName(graph.node(relu).output, "C");
    graph.addOutput(graph.node(relu).output);

    forgecc::lowering::LoopLowerer lowerer;
    auto program = lowerer.lower(graph);
    assert(program.has_value());
    return std::move(*program);
}

void testLoopProgramEmitterUsesLoopIr() {
    const auto program = lowerMatmulRelu();
    forgecc::codegen::CppEmitter emitter;
    forgecc::common::DiagnosticEngine diagnostics;
    const auto source = emitter.generate(program, diagnostics);

    assert(source.has_value());
    assert(!diagnostics.hasErrors());
    assert(source->find("for (std::size_t i = 0; i < 2; i += 1)") != std::string::npos);
    assert(source->find("for (std::size_t j = 0; j < 2; j += 1)") != std::string::npos);
    assert(source->find("for (std::size_t k = 0; k < 3; k += 1)") != std::string::npos);
    assert(source->find("input_0.data") != std::string::npos);
    assert(source->find("input_1.data") != std::string::npos);
    assert(source->find("value_2.data") != std::string::npos);
    assert(source->find("value_3.data") != std::string::npos);
    assert(source->find("value_3.data[(i0) * 2 + (i1)]") != std::string::npos);
    assert(source->find("((value_2.data[(i0) * 2 + (i1)]) > 0.0F") !=
           std::string::npos);
    assert(source == emitter.generate(program, diagnostics));
}

void testStandaloneEmitterProvidesExecutableWrapper() {
    const auto program = lowerRelu();
    forgecc::codegen::StandaloneEmitter emitter;
    forgecc::common::DiagnosticEngine diagnostics;
    const auto source = emitter.generate(program, diagnostics);

    assert(source.has_value());
    assert(source->find("int main()") != std::string::npos);
    assert(source->find("input_0.data.resize(4)") != std::string::npos);
    assert(source->find("forgecc_generated::compute(input_0, output_1);") !=
           std::string::npos);
    assert(source->find("std::cout << output_1.data[index]") != std::string::npos);
}

void testStandaloneSourceCompilesToExecutable() {
    const auto program = lowerRelu();
    forgecc::codegen::StandaloneEmitter emitter;
    forgecc::common::DiagnosticEngine diagnostics;
    const auto source = emitter.generate(program, diagnostics);
    assert(source.has_value());

    const auto output = std::filesystem::temp_directory_path() / "forgecc loop standalone test";
    std::error_code error;
    std::filesystem::remove(output, error);
    forgecc::native::NativeCompiler compiler;
    const auto result = compiler.compile(*source, output);
    assert(result.success);
    assert(std::filesystem::is_regular_file(output));
    std::filesystem::remove(output, error);
}

void testLoopProgramEmitterRejectsUnknownTensor() {
    auto program = lowerRelu();
    program.tensorNames.at(0) = "renamed";

    forgecc::codegen::CppEmitter emitter;
    forgecc::common::DiagnosticEngine diagnostics;
    assert(!emitter.generate(program, diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

} // namespace

void runStandaloneEmitterTests() {
    testLoopProgramEmitterUsesLoopIr();
    testStandaloneEmitterProvidesExecutableWrapper();
    testStandaloneSourceCompilesToExecutable();
    testLoopProgramEmitterRejectsUnknownTensor();
}
