#include "forgecc/optimizer/common_subexpression_elimination.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

using forgecc::common::DiagnosticEngine;
using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::optimizer::CommonSubexpressionElimination;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

TensorType matrix(std::size_t rows = 2, std::size_t columns = 2) {
    return TensorType(ElementType::F32, {rows, columns});
}

void testDuplicateReluIsMerged() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto canonical = graph.addNode(OperationKind::Relu, {input}, matrix());
    static_cast<void>(graph.addNode(OperationKind::Relu, {input}, matrix()));
    graph.addOutput(graph.node(canonical).output);

    DiagnosticEngine diagnostics;
    const auto result = CommonSubexpressionElimination{}.run(graph, diagnostics);
    assert(result.changed);
    assert(graph.nodeCount() == 1);
    assert(graph.validate(diagnostics));
}

void testDuplicateMatmulWithOrderedInputsIsMerged() {
    Graph graph;
    const auto a = graph.addInput("A", matrix(2, 3));
    const auto b = graph.addInput("B", matrix(3, 4));
    const auto canonical = graph.addNode(OperationKind::Matmul, {a, b}, matrix(2, 4));
    static_cast<void>(graph.addNode(OperationKind::Matmul, {a, b}, matrix(2, 4)));
    graph.addOutput(graph.node(canonical).output);

    DiagnosticEngine diagnostics;
    assert(CommonSubexpressionElimination{}.run(graph, diagnostics).changed);
    assert(graph.nodeCount() == 1);
}

void testDifferentOperationsAndInputsAreNotMerged() {
    Graph graph;
    const auto a = graph.addInput("A", matrix());
    const auto b = graph.addInput("B", matrix());
    const auto relu = graph.addNode(OperationKind::Relu, {a}, matrix());
    const auto matmul = graph.addNode(OperationKind::Matmul, {a, b}, matrix());
    const auto reverseA = graph.addInput("C", matrix());
    const auto reverseB = graph.addInput("D", matrix());
    const auto firstMatmul =
        graph.addNode(OperationKind::Matmul, {reverseA, reverseB}, matrix());
    const auto reversedMatmul =
        graph.addNode(OperationKind::Matmul, {reverseB, reverseA}, matrix());
    graph.addOutput(graph.node(relu).output);
    graph.addOutput(graph.node(matmul).output);
    graph.addOutput(graph.node(firstMatmul).output);
    graph.addOutput(graph.node(reversedMatmul).output);

    DiagnosticEngine diagnostics;
    const auto result = CommonSubexpressionElimination{}.run(graph, diagnostics);
    assert(!result.changed);
    assert(graph.nodeCount() == 4);
}

void testConsumersAreRedirected() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto canonical = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto duplicate = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto consumer = graph.addNode(OperationKind::Relu, {graph.node(duplicate).output},
                                        matrix());
    graph.addOutput(graph.node(consumer).output);

    DiagnosticEngine diagnostics;
    assert(CommonSubexpressionElimination{}.run(graph, diagnostics).changed);
    assert(graph.nodeCount() == 2);
    assert(graph.node(consumer).inputs == std::vector<forgecc::ir::ValueId>({
                                                graph.node(canonical).output}));
    assert(graph.validate(diagnostics));
}

void testMultipleConsumersAreRedirected() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto canonical = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto duplicate = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto left = graph.addNode(OperationKind::Relu, {graph.node(duplicate).output}, matrix());
    const auto right = graph.addNode(OperationKind::Relu, {graph.node(duplicate).output}, matrix());
    graph.addOutput(graph.node(left).output);
    graph.addOutput(graph.node(right).output);

    DiagnosticEngine diagnostics;
    assert(CommonSubexpressionElimination{}.run(graph, diagnostics).changed);
    assert(graph.node(left).inputs[0] == graph.node(canonical).output);
    assert(graph.node(right).inputs[0] == graph.node(canonical).output);
    assert(graph.validate(diagnostics));
}

void testNamedGraphOutputsArePreservedBySkippingUnsafeMerge() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto left = graph.addNode(OperationKind::Relu, {input}, matrix());
    const auto right = graph.addNode(OperationKind::Relu, {input}, matrix());
    graph.setValueName(graph.node(left).output, "left");
    graph.setValueName(graph.node(right).output, "right");
    graph.addOutput(graph.node(left).output);
    graph.addOutput(graph.node(right).output);

    DiagnosticEngine diagnostics;
    const auto result = CommonSubexpressionElimination{}.run(graph, diagnostics);
    assert(!result.changed);
    assert(graph.nodeCount() == 2);
    assert(graph.value(graph.outputs()[0]).name == "left");
    assert(graph.value(graph.outputs()[1]).name == "right");
}

void testRepeatedRunsAreIdempotent() {
    Graph graph;
    const auto input = graph.addInput("A", matrix());
    const auto canonical = graph.addNode(OperationKind::Relu, {input}, matrix());
    static_cast<void>(graph.addNode(OperationKind::Relu, {input}, matrix()));
    graph.addOutput(graph.node(canonical).output);

    DiagnosticEngine diagnostics;
    CommonSubexpressionElimination pass;
    assert(pass.run(graph, diagnostics).changed);
    assert(!pass.run(graph, diagnostics).changed);
    assert(graph.validate(diagnostics));
}

} // namespace

void runCommonSubexpressionEliminationTests() {
    testDuplicateReluIsMerged();
    testDuplicateMatmulWithOrderedInputsIsMerged();
    testDifferentOperationsAndInputsAreNotMerged();
    testConsumersAreRedirected();
    testMultipleConsumersAreRedirected();
    testNamedGraphOutputsArePreservedBySkippingUnsafeMerge();
    testRepeatedRunsAreIdempotent();
}
