#include "forgecc/ir/graph.hpp"

#include <cassert>
#include <cstddef>
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

} // namespace

void runGraphTests() {
    testGraphInputsAndMatmul();
    testOperationResultConnections();
    testValueAndNodeIdsRemainUsableAsGraphGrows();
}
