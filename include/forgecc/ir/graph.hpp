#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/node.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace forgecc::ir {

class Graph {
public:
    [[nodiscard]] ValueId addInput(std::string name, semantic::TensorType type);
    [[nodiscard]] NodeId addNode(OperationKind operation, std::vector<ValueId> inputs,
                                  semantic::TensorType outputType);
    void addOutput(ValueId id);
    void setValueName(ValueId id, std::string name);

    [[nodiscard]] const Value& value(ValueId id) const;
    [[nodiscard]] const Node& node(NodeId id) const;
    [[nodiscard]] std::size_t valueCount() const;
    [[nodiscard]] std::size_t nodeCount() const;
    [[nodiscard]] const std::vector<ValueId>& outputs() const;
    [[nodiscard]] bool validate(common::DiagnosticEngine& diagnostics) const;
    [[nodiscard]] std::string dump() const;

private:
    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<ValueId> outputs_;
};

} // namespace forgecc::ir
