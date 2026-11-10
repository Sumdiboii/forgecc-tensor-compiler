#include "forgecc/ir/graph.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <string>
#include <utility>

namespace forgecc::ir {

namespace {

std::string operationName(OperationKind operation) {
    switch (operation) {
    case OperationKind::Relu:
        return "Relu";
    case OperationKind::Matmul:
        return "Matmul";
    case OperationKind::Add:
        return "Add";
    case OperationKind::Subtract:
        return "Subtract";
    case OperationKind::Multiply:
        return "Multiply";
    case OperationKind::Divide:
        return "Divide";
    }

    return "Unknown";
}

std::size_t expectedArity(OperationKind operation) {
    switch (operation) {
    case OperationKind::Relu:
        return 1;
    case OperationKind::Matmul:
        return 2;
    case OperationKind::Add:
    case OperationKind::Subtract:
    case OperationKind::Multiply:
    case OperationKind::Divide:
        return 2;
    }

    return 0;
}

std::string tensorTypeName(const semantic::TensorType& type) {
    std::string result = type.elementType == semantic::ElementType::F32 ? "f32[" : "?[";
    for (std::size_t index = 0; index < type.dimensions.size(); ++index) {
        if (index != 0) {
            result += ", ";
        }
        result += std::to_string(type.dimensions[index]);
    }
    result += ']';
    return result;
}

std::string valueReference(ValueId id) {
    return "%" + std::to_string(id);
}

} // namespace

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
    nodes_.push_back(Node{id, operation, std::move(inputs), output, true});
    return id;
}

void Graph::addOutput(ValueId id) {
    static_cast<void>(value(id));
    if (std::find(outputs_.begin(), outputs_.end(), id) == outputs_.end()) {
        outputs_.push_back(id);
    }
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

bool Graph::isNodeLive(NodeId id) const {
    return nodes_.at(id).isLive;
}

std::size_t Graph::valueCount() const {
    return values_.size();
}

std::size_t Graph::nodeCount() const {
    return static_cast<std::size_t>(std::count_if(
        nodes_.begin(), nodes_.end(), [](const Node& node) { return node.isLive; }));
}

const std::vector<ValueId>& Graph::outputs() const {
    return outputs_;
}

bool Graph::validate(common::DiagnosticEngine& diagnostics) const {
    bool valid = true;
    const auto report = [&diagnostics, &valid](std::string message) {
        diagnostics.reportError({1, 1}, std::move(message));
        valid = false;
    };

    for (std::size_t index = 0; index < values_.size(); ++index) {
        const Value& value = values_[index];
        if (value.id != index) {
            report("value ID does not match its position in the graph");
        }

        if (value.kind == ValueKind::GraphInput) {
            if (value.producer.has_value()) {
                report("graph input has an unexpected producer");
            }
        } else if (value.kind == ValueKind::OperationOutput) {
            if (!value.producer.has_value()) {
                report("operation output is missing its producer");
            } else if (*value.producer >= nodes_.size()) {
                report("operation output refers to an invalid producer node");
            } else if (!nodes_[*value.producer].isLive) {
                continue;
            } else if (nodes_[*value.producer].output != value.id) {
                report("operation output producer does not reference the value");
            }
        }
    }

    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        const Node& node = nodes_[index];
        if (!node.isLive) {
            continue;
        }
        if (node.id != index) {
            report("node ID does not match its position in the graph");
        }

        if (node.inputs.size() != expectedArity(node.operation)) {
            report("" + operationName(node.operation) + " node has incorrect input arity");
        }

        if (node.output >= values_.size()) {
            report("node output refers to an invalid value");
        } else {
            const Value& output = values_[node.output];
            if (output.kind != ValueKind::OperationOutput || !output.producer.has_value() ||
                *output.producer != node.id) {
                report("node output has an inconsistent producer relationship");
            }
        }

        for (ValueId input : node.inputs) {
            if (input >= values_.size()) {
                report("node input refers to an invalid value");
                continue;
            }

            const Value& inputValue = values_[input];
            if (inputValue.kind == ValueKind::OperationOutput) {
                if (!inputValue.producer.has_value() || *inputValue.producer >= nodes_.size() ||
                    !nodes_[*inputValue.producer].isLive ||
                    nodes_[*inputValue.producer].output != input) {
                    report("operation input has an inconsistent producer relationship");
                }
            }
        }
    }

    for (ValueId output : outputs_) {
        if (output >= values_.size()) {
            report("graph output refers to an invalid value");
        } else if (values_[output].kind == ValueKind::OperationOutput &&
                   (!values_[output].producer.has_value() ||
                    *values_[output].producer >= nodes_.size() ||
                    !nodes_[*values_[output].producer].isLive)) {
            report("graph output refers to a removed node result");
        }
    }

    return valid;
}

std::optional<std::vector<NodeId>> Graph::topologicalOrder(
    common::DiagnosticEngine& diagnostics) const {
    if (!validate(diagnostics)) {
        return std::nullopt;
    }

    std::vector<std::size_t> indegree(nodes_.size(), 0);
    std::vector<std::vector<NodeId>> consumers(nodes_.size());
    for (const Node& node : nodes_) {
        if (!node.isLive) {
            continue;
        }
        for (ValueId input : node.inputs) {
            const Value& value = values_[input];
            if (value.kind != ValueKind::OperationOutput) {
                continue;
            }

            const NodeId producer = *value.producer;
            ++indegree[node.id];
            consumers[producer].push_back(node.id);
        }
    }

    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> ready;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        if (nodes_[id].isLive && indegree[id] == 0) {
            ready.push(id);
        }
    }

    std::vector<NodeId> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
        const NodeId current = ready.top();
        ready.pop();
        order.push_back(current);

        for (NodeId consumer : consumers[current]) {
            --indegree[consumer];
            if (indegree[consumer] == 0) {
                ready.push(consumer);
            }
        }
    }

    if (order.size() != nodeCount()) {
        diagnostics.reportError({1, 1}, "computation graph contains a cycle");
        return std::nullopt;
    }

    return order;
}

bool Graph::replaceValueInConsumers(ValueId from, ValueId to) {
    static_cast<void>(value(from));
    static_cast<void>(value(to));

    bool changed = false;
    for (Node& node : nodes_) {
        if (!node.isLive) {
            continue;
        }
        for (ValueId& input : node.inputs) {
            if (input == from) {
                input = to;
                changed = true;
            }
        }
    }
    return changed;
}

bool Graph::keepOnlyNodes(const std::vector<NodeId>& liveNodes) {
    std::vector<bool> keep(nodes_.size(), false);
    for (NodeId id : liveNodes) {
        if (id >= nodes_.size()) {
            return false;
        }
        keep[id] = true;
    }

    bool changed = false;
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        if (nodes_[index].isLive && !keep[index]) {
            nodes_[index].isLive = false;
            changed = true;
        }
    }
    return changed;
}

std::string Graph::dump() const {
    std::string result = "# ForgeCC Computation Graph\nInputs:\n";
    for (const Value& value : values_) {
        if (value.kind != ValueKind::GraphInput) {
            continue;
        }
        result += valueReference(value.id) + " = " + value.name + " : " +
                  tensorTypeName(value.type) + "\n";
    }

    result += "\nNodes:\n";
    for (const Node& node : nodes_) {
        if (!node.isLive) {
            continue;
        }
        result += "node" + std::to_string(node.id) + " = " + operationName(node.operation) + "(";
        for (std::size_t index = 0; index < node.inputs.size(); ++index) {
            if (index != 0) {
                result += ", ";
            }
            result += valueReference(node.inputs[index]);
        }
        result += ") -> " + valueReference(node.output);
        if (node.output < values_.size() && !values_[node.output].name.empty()) {
            result += " (" + values_[node.output].name + ")";
        }
        if (node.output < values_.size()) {
            result += " : " + tensorTypeName(values_[node.output].type);
        }
        result += "\n";
    }

    result += "\nOutputs:\n";
    for (ValueId output : outputs_) {
        result += valueReference(output);
        if (output < values_.size() && !values_[output].name.empty()) {
            result += " = " + values_[output].name;
        }
        result += "\n";
    }

    return result;
}

} // namespace forgecc::ir
