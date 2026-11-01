#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace forgecc::semantic {

enum class ElementType {
    F32,
};

struct TensorType {
    ElementType elementType;
    std::vector<std::size_t> dimensions;

    TensorType(ElementType elementType, std::vector<std::size_t> dimensions)
        : elementType(elementType), dimensions(std::move(dimensions)) {}
};

} // namespace forgecc::semantic
