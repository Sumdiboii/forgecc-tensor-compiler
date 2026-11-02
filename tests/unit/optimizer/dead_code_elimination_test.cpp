#include "forgecc/optimizer/dead_code_elimination.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

using forgecc::common::DiagnosticEngine;
using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::optimizer::DeadCodeElimination;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

TensorType matrix() {
    return TensorType(ElementType::F32, {2, 2});
}

void testDeadNodeIsRemoved() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    static_cast<void>(graph.addNode(OperationKind::Relu, {input}, matrix()));

    DiagnosticEngine diagnostics;
    const auto result = DeadCodeElimination{}.run(graph, diagnostics);
    assert(result.changed);
    assert(graph.nodeCount() == 0);
    assert(graph.validate(diagnostics));
}

void testDeadChainIsRemoved() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto first = graph.addNode(OperationKind::Relu, {input}, matrix());
    static_cast<void>(graph.addNode(OperationKind::Relu, {graph.node(first).output}, matrix()));

    DiagnosticEngine diagnostics;
    const auto result = DeadCodeElimination{}.run(graph, diagnostics);
    assert(result.changed);
    assert(graph.nodeCount() == 0);
    assert(graph.outputs().empty());
}

void testLiveDependenciesArePreserved() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto first = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto output = graph.addNode(OperationKind::Relu, {graph.node(first).output}, matrix());
    graph.setValueName(graph.node(output).output, "out");
    graph.addOutput(graph.node(output).output);

    DiagnosticEngine diagnostics;
    const auto result = DeadCodeElimination{}.run(graph, diagnostics);
    assert(!result.changed);
    assert(graph.nodeCount() == 2);
    assert(graph.value(graph.outputs().front()).name == "out");
}

void testMultipleOutputsPreserveSeparateDependencies() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto first = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto second = graph.addNode(OperationKind::Relu, {input}, matrix());
    graph.setValueName(graph.node(first).output, "left");
    graph.setValueName(graph.node(second).output, "right");
    graph.addOutput(graph.node(first).output);
    graph.addOutput(graph.node(second).output);

    DiagnosticEngine diagnostics;
    const auto result = DeadCodeElimination{}.run(graph, diagnostics);
    assert(!result.changed);
    assert(graph.nodeCount() == 2);
    assert(graph.value(graph.outputs()[0]).name == "left");
    assert(graph.value(graph.outputs()[1]).name == "right");
}

void testNoOutputsRemoveAllNodes() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    static_cast<void>(graph.addNode(OperationKind::Relu, {input}, matrix()));

    DiagnosticEngine diagnostics;
    const auto result = DeadCodeElimination{}.run(graph, diagnostics);
    assert(result.changed);
    assert(graph.nodeCount() == 0);
}

void testRepeatedRunsAreStable() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto output = graph.addNode(OperationKind::Relu, {input}, matrix());
    graph.addOutput(graph.node(output).output);

    DiagnosticEngine diagnostics;
    DeadCodeElimination pass;
    assert(!pass.run(graph, diagnostics).changed);
    assert(!pass.run(graph, diagnostics).changed);
    assert(graph.validate(diagnostics));
}

} // namespace

void runDeadCodeEliminationTests() {
    testDeadNodeIsRemoved();
    testDeadChainIsRemoved();
    testLiveDependenciesArePreserved();
    testMultipleOutputsPreserveSeparateDependencies();
    testNoOutputsRemoveAllNodes();
    testRepeatedRunsAreStable();
}
