#pragma once

#include "forgecc/semantic/type.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace forgecc::ir {

using ValueId = std::size_t;
using NodeId = std::size_t;

enum class ValueKind {
    GraphInput,
    OperationOutput,
};

struct Value {
    ValueId id;
    ValueKind kind;
    std::string name;
    semantic::TensorType type;
    std::optional<NodeId> producer;
};

} // namespace forgecc::ir
