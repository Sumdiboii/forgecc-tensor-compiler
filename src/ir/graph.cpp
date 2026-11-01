#include "forgecc/ir/graph.hpp"

#include <utility>

namespace forgecc::ir {

ValueId Graph::addInput(std::string name, semantic::TensorType type) {
    const ValueId id = values_.size();
    values_.push_back(Value{id, ValueKind::GraphInput, std::move(name), std::move(type),
                            std::nullopt});
    return id;
}

NodeId Graph::addNode(OperationKind operation, std::vector<ValueId> inputs,
                      semantic::TensorType outputType) {
    const NodeId id = nodes_.size();
    const ValueId output = values_.size();
    values_.push_back(
        Value{output, ValueKind::OperationOutput, {}, std::move(outputType), id});
    nodes_.push_back(Node{id, operation, std::move(inputs), output});
    return id;
}

void Graph::setValueName(ValueId id, std::string name) {
    values_.at(id).name = std::move(name);
}

const Value& Graph::value(ValueId id) const {
    return values_.at(id);
}

const Node& Graph::node(NodeId id) const {
    return nodes_.at(id);
}

std::size_t Graph::valueCount() const {
    return values_.size();
}

std::size_t Graph::nodeCount() const {
    return nodes_.size();
}

} // namespace forgecc::ir
