#pragma once

#include "forgecc/ir/value.hpp"

#include <vector>

namespace forgecc::ir {

enum class OperationKind {
    Relu,
    Matmul,
    Add,
    Subtract,
    Multiply,
    Divide,
};

struct Node {
    NodeId id;
    OperationKind operation;
    std::vector<ValueId> inputs;
    ValueId output;
    bool isLive = true;
};

} // namespace forgecc::ir
