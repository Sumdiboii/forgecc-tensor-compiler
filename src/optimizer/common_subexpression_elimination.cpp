#include "forgecc/optimizer/common_subexpression_elimination.hpp"

#include <algorithm>
#include <unordered_set>

namespace forgecc::optimizer {

namespace {

bool sameType(const semantic::TensorType& left, const semantic::TensorType& right) {
    return left.elementType == right.elementType && left.dimensions == right.dimensions;
}

bool sameComputation(const ir::Node& left, const ir::Node& right, const ir::Graph& graph) {
    if (left.operation != right.operation || left.inputs != right.inputs) {
        return false;
    }

    return sameType(graph.value(left.output).type, graph.value(right.output).type);
}

bool isGraphOutput(const ir::Graph& graph, ir::ValueId value) {
    const auto& outputs = graph.outputs();
    return std::find(outputs.begin(), outputs.end(), value) != outputs.end();
}

} // namespace

std::string_view CommonSubexpressionElimination::name() const {
    return "common-subexpression-elimination";
}

PassResult CommonSubexpressionElimination::run(ir::Graph& graph,
                                               common::DiagnosticEngine& diagnostics) {
    const auto order = graph.topologicalOrder(diagnostics);
    if (!order.has_value()) {
        return PassResult{};
    }

    std::vector<ir::NodeId> canonicalNodes;
    std::unordered_set<ir::NodeId> redundantNodes;
    bool changed = false;

    for (ir::NodeId nodeId : *order) {
        if (redundantNodes.contains(nodeId)) {
            continue;
        }

        const ir::Node& node = graph.node(nodeId);
        ir::NodeId canonicalId = nodeId;
        for (ir::NodeId candidateId : canonicalNodes) {
            if (sameComputation(node, graph.node(candidateId), graph)) {
                canonicalId = candidateId;
                break;
            }
        }

        if (canonicalId == nodeId) {
            canonicalNodes.push_back(nodeId);
            continue;
        }

        const ir::Value& redundantValue = graph.value(node.output);
        const ir::Value& canonicalValue = graph.value(graph.node(canonicalId).output);
        if (isGraphOutput(graph, node.output) ||
            (!redundantValue.name.empty() && !canonicalValue.name.empty() &&
             redundantValue.name != canonicalValue.name)) {
            canonicalNodes.push_back(nodeId);
            continue;
        }

        if (canonicalValue.name.empty() && !redundantValue.name.empty()) {
            graph.setValueName(graph.node(canonicalId).output, redundantValue.name);
        }
        static_cast<void>(
            graph.replaceValueInConsumers(node.output, graph.node(canonicalId).output));
        redundantNodes.insert(nodeId);
        changed = true;
    }

    if (!changed) {
        return PassResult{};
    }

    std::vector<ir::NodeId> survivingNodes;
    survivingNodes.reserve(order->size() - redundantNodes.size());
    for (ir::NodeId nodeId : *order) {
        if (!redundantNodes.contains(nodeId)) {
            survivingNodes.push_back(nodeId);
        }
    }
    if (!graph.keepOnlyNodes(survivingNodes) || !graph.validate(diagnostics)) {
        return PassResult{};
    }

    return PassResult{true};
}

} // namespace forgecc::optimizer
