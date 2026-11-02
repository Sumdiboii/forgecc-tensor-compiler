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
    graph.addOutput(graph.node(node).output);
    CppEmitter emitter(CodegenOptions{"test_generated"});
    DiagnosticEngine diagnostics;

    const auto source = emitter.generate(graph, diagnostics);
    assert(source.has_value());
    assert(!diagnostics.hasErrors());
    assert(source->find("namespace test_generated {") != std::string::npos);
    assert(source->find("Graph values: 2.") != std::string::npos);
    assert(source->find("Graph operations: 1.") != std::string::npos);
    assert(source == emitter.generate(graph, diagnostics));
}

void testCppEmitterRejectsInvalidGraphs() {
    Graph graph;
    static_cast<void>(graph.addNode(OperationKind::Relu, {42}, matrix()));
    CppEmitter emitter;
    DiagnosticEngine diagnostics;

    assert(!emitter.generate(graph, diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

} // namespace

void runCppEmitterTests() {
    testSourceEmitterProducesLinesAndBlankLines();
    testSourceEmitterOutputIsDeterministic();
    testCppEmitterGeneratesDeterministicSkeleton();
    testCppEmitterRejectsInvalidGraphs();
}
