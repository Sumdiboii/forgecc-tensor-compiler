#include "forgecc/codegen/cpp_emitter.hpp"

#include <cassert>
#include <string>

namespace {

using forgecc::codegen::CodegenOptions;
using forgecc::codegen::CppEmitter;
using forgecc::codegen::SourceEmitter;
using forgecc::common::DiagnosticEngine;
using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

TensorType matrix() {
    return TensorType(ElementType::F32, {2, 2});
}

void testSourceEmitterProducesLinesAndBlankLines() {
    SourceEmitter emitter;
    emitter.emitLine("first");
    emitter.emitBlankLine();
    emitter.emitLine("second");

    assert(emitter.str() == "first\n\nsecond\n");
}

void testSourceEmitterOutputIsDeterministic() {
    SourceEmitter first;
    SourceEmitter second;
    for (SourceEmitter* emitter : {&first, &second}) {
        emitter->emitLine("namespace forgecc_generated {");
        emitter->emitLine("}");
    }

    assert(first.str() == second.str());
}

void testCppEmitterGeneratesDeterministicSkeleton() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto node = graph.addNode(OperationKind::Relu, {input}, matrix());
    graph.setValueName(graph.node(node).output, "out");
    graph.addOutput(graph.node(node).output);
    CppEmitter emitter(CodegenOptions{"test_generated"});
    DiagnosticEngine diagnostics;

    const auto source = emitter.generate(graph, diagnostics);
    assert(source.has_value());
    assert(!diagnostics.hasErrors());
    assert(source->find("namespace test_generated {") != std::string::npos);
    assert(source->find("struct Tensor {") != std::string::npos);
    assert(source->find("// input A -> input_0") != std::string::npos);
    assert(source->find("// output out -> output_1") != std::string::npos);
    assert(source->find("void compute(const Tensor& input_0, Tensor& output_1) {") !=
           std::string::npos);
    assert(source->find("value_1.data[index]") != std::string::npos);
    assert(source->find("output_1 = value_1;") != std::string::npos);
    assert(source == emitter.generate(graph, diagnostics));
}

void testCppEmitterUsesTopologicalOrderForChainedOperations() {
    Graph graph;
    const auto a = graph.addInput("A", TensorType(ElementType::F32, {2, 3}));
    const auto b = graph.addInput("B", TensorType(ElementType::F32, {3, 4}));
    const auto matmul = graph.addNode(OperationKind::Matmul, {a, b},
                                      TensorType(ElementType::F32, {2, 4}));
    const auto relu = graph.addNode(OperationKind::Relu, {graph.node(matmul).output},
                                    TensorType(ElementType::F32, {2, 4}));
    graph.setValueName(graph.node(relu).output, "out");
    graph.addOutput(graph.node(relu).output);
    CppEmitter emitter;
    DiagnosticEngine diagnostics;

    const auto source = emitter.generate(graph, diagnostics);
    assert(source.has_value());
    const std::size_t matmulPosition = source->find("float sum = 0.0F;");
    const std::size_t reluPosition = source->find("value_3.data[index]");
    assert(matmulPosition != std::string::npos);
    assert(reluPosition != std::string::npos);
    assert(matmulPosition < reluPosition);
    assert(source->find("input_0.data[row * 3 + inner]") != std::string::npos);
    assert(source->find("input_1.data[inner * 4 + column]") != std::string::npos);
    assert(source->find("// output out -> output_3") != std::string::npos);
    assert(source->find("output_3 = value_3;") != std::string::npos);
}

void testCppEmitterRejectsIncompatibleMatmulShapes() {
    Graph graph;
    const auto a = graph.addInput("A", TensorType(ElementType::F32, {2, 3}));
    const auto b = graph.addInput("B", TensorType(ElementType::F32, {4, 2}));
    const auto result = graph.addNode(OperationKind::Matmul, {a, b},
                                      TensorType(ElementType::F32, {2, 2}));
    graph.addOutput(graph.node(result).output);
    CppEmitter emitter;
    DiagnosticEngine diagnostics;

    assert(!emitter.generate(graph, diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

void testCppEmitterRejectsInvalidGraphs() {
    Graph graph;
    static_cast<void>(graph.addNode(OperationKind::Relu, {42}, matrix()));
    CppEmitter emitter;
    DiagnosticEngine diagnostics;

    assert(!emitter.generate(graph, diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

void testCppEmitterRejectsInvalidNamespaceNames() {
    Graph graph;
    CppEmitter emitter(CodegenOptions{"bad-namespace"});
    DiagnosticEngine diagnostics;

    assert(!emitter.generate(graph, diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

void testCppEmitterSupportsElementwiseBinaryOperations() {
    Graph graph;
    const auto left = graph.addInput("A", matrix());
    const auto right = graph.addInput("B", matrix());
    const auto node = graph.addNode(OperationKind::Add, {left, right}, matrix());
    graph.addOutput(graph.node(node).output);
    CppEmitter emitter;
    DiagnosticEngine diagnostics;

    const auto source = emitter.generate(graph, diagnostics);
    assert(source.has_value());
    assert(!diagnostics.hasErrors());
    assert(source->find("input_0.data[index] + input_1.data[index]") != std::string::npos);
}

} // namespace

void runCppEmitterTests() {
    testSourceEmitterProducesLinesAndBlankLines();
    testSourceEmitterOutputIsDeterministic();
    testCppEmitterGeneratesDeterministicSkeleton();
    testCppEmitterUsesTopologicalOrderForChainedOperations();
    testCppEmitterRejectsIncompatibleMatmulShapes();
    testCppEmitterRejectsInvalidGraphs();
    testCppEmitterRejectsInvalidNamespaceNames();
    testCppEmitterSupportsElementwiseBinaryOperations();
}
