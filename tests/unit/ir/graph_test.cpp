#include "forgecc/ir/graph.hpp"

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace {

using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::ir::ValueKind;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

TensorType matrix(std::size_t rows, std::size_t columns) {
    return TensorType(ElementType::F32, {rows, columns});
}

void testGraphInputsAndMatmul() {
    Graph graph;
    const auto a = graph.addInput("A", matrix(2, 3));
    const auto b = graph.addInput("B", matrix(3, 4));
    const auto matmul = graph.addNode(OperationKind::Matmul, {a, b}, matrix(2, 4));

    assert(graph.valueCount() == 3);
    assert(graph.nodeCount() == 1);
    assert(graph.value(a).kind == ValueKind::GraphInput);
    assert(graph.value(a).name == "A");
    assert(graph.value(a).type.dimensions == std::vector<std::size_t>({2, 3}));
    assert(graph.node(matmul).operation == OperationKind::Matmul);
    assert(graph.node(matmul).inputs == std::vector<forgecc::ir::ValueId>({a, b}));

    const auto output = graph.node(matmul).output;
    assert(graph.value(output).kind == ValueKind::OperationOutput);
    assert(graph.value(output).producer == matmul);
    assert(graph.value(output).type.dimensions == std::vector<std::size_t>({2, 4}));
}

void testOperationResultConnections() {
    Graph graph;
    const auto input = graph.addInput("A", matrix(2, 2));
    const auto matmul = graph.addNode(OperationKind::Matmul, {input, input}, matrix(2, 2));
    const auto matmulOutput = graph.node(matmul).output;
    const auto relu = graph.addNode(OperationKind::Relu, {matmulOutput}, matrix(2, 2));

    assert(graph.node(relu).operation == OperationKind::Relu);
    assert(graph.node(relu).inputs.size() == 1);
    assert(graph.node(relu).inputs[0] == matmulOutput);
    assert(graph.value(graph.node(relu).output).producer == relu);
}

void testValueAndNodeIdsRemainUsableAsGraphGrows() {
    Graph graph;
    const auto input = graph.addInput("A", matrix(1, 1));
    const auto firstNode = graph.addNode(OperationKind::Relu, {input}, matrix(1, 1));

    for (int index = 0; index < 32; ++index) {
        static_cast<void>(graph.addInput("extra", matrix(1, 1)));
    }

    assert(graph.value(input).name == "A");
    assert(graph.node(firstNode).operation == OperationKind::Relu);
    assert(graph.node(firstNode).inputs[0] == input);
}

void testValidGraphPassesValidation() {
    Graph graph;
    const auto a = graph.addInput("A", matrix(2, 3));
    const auto b = graph.addInput("B", matrix(3, 4));
    const auto matmul = graph.addNode(OperationKind::Matmul, {a, b}, matrix(2, 4));
    graph.setValueName(graph.node(matmul).output, "C");
    graph.addOutput(graph.node(matmul).output);

    forgecc::common::DiagnosticEngine diagnostics;
    assert(graph.validate(diagnostics));
    assert(!diagnostics.hasErrors());
}

void testInvalidNodeInputFailsValidation() {
    Graph graph;
    static_cast<void>(graph.addNode(OperationKind::Relu, {99}, matrix(2, 2)));

    forgecc::common::DiagnosticEngine diagnostics;
    assert(!graph.validate(diagnostics));
    assert(diagnostics.hasErrors());
}

void testInvalidOperationArityFailsValidation() {
    Graph graph;
    static_cast<void>(graph.addNode(OperationKind::Relu, {}, matrix(2, 2)));

    forgecc::common::DiagnosticEngine diagnostics;
    assert(!graph.validate(diagnostics));
    assert(diagnostics.hasErrors());
}

void testGraphDumpIsReadableAndDeterministic() {
    Graph graph;
    const auto a = graph.addInput("A", matrix(2, 3));
    const auto b = graph.addInput("B", matrix(3, 4));
    const auto matmul = graph.addNode(OperationKind::Matmul, {a, b}, matrix(2, 4));
    const auto relu = graph.addNode(OperationKind::Relu, {graph.node(matmul).output},
                                    matrix(2, 4));
    graph.setValueName(graph.node(relu).output, "out");
    graph.addOutput(graph.node(relu).output);

    const std::string firstDump = graph.dump();
    assert(firstDump == graph.dump());
    assert(firstDump.find("A : f32[2, 3]") != std::string::npos);
    assert(firstDump.find("B : f32[3, 4]") != std::string::npos);
    assert(firstDump.find("Matmul(%0, %1) -> %2") != std::string::npos);
    assert(firstDump.find("Relu(%2) -> %3 (out) : f32[2, 4]") != std::string::npos);
    assert(firstDump.find("%3 = out") != std::string::npos);
}

void testTopologicalOrderIsDeterministicAndDependencyRespecting() {
    Graph graph;
    static_cast<void>(graph.addInput("A", matrix(2, 2)));
    static_cast<void>(graph.addInput("B", matrix(2, 2)));
    const auto laterOutput = graph.addNode(OperationKind::Relu, {3}, matrix(2, 2));
    const auto earlierOutput = graph.addNode(OperationKind::Relu, {0}, matrix(2, 2));
    graph.addOutput(graph.node(laterOutput).output);
    graph.addOutput(graph.node(earlierOutput).output);

    forgecc::common::DiagnosticEngine diagnostics;
    const auto order = graph.topologicalOrder(diagnostics);
    assert(order.has_value());
    assert(*order == std::vector<forgecc::ir::NodeId>({1, 0}));
    assert(!diagnostics.hasErrors());
}

void testTopologicalOrderDetectsCycles() {
    Graph graph;
    static_cast<void>(graph.addInput("A", matrix(2, 2)));
    static_cast<void>(graph.addInput("B", matrix(2, 2)));
    const auto first = graph.addNode(OperationKind::Relu, {3}, matrix(2, 2));
    const auto second = graph.addNode(OperationKind::Relu, {graph.node(first).output},
                                      matrix(2, 2));
    static_cast<void>(second);

    forgecc::common::DiagnosticEngine diagnostics;
    assert(!graph.topologicalOrder(diagnostics).has_value());
    assert(diagnostics.hasErrors());
}

} // namespace

void runGraphTests() {
    testGraphInputsAndMatmul();
    testOperationResultConnections();
    testValueAndNodeIdsRemainUsableAsGraphGrows();
    testValidGraphPassesValidation();
    testInvalidNodeInputFailsValidation();
    testInvalidOperationArityFailsValidation();
    testGraphDumpIsReadableAndDeterministic();
    testTopologicalOrderIsDeterministicAndDependencyRespecting();
    testTopologicalOrderDetectsCycles();
}
