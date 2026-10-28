#pragma once

#include <cstddef>

namespace forgecc::common {

struct SourceLocation {
    std::size_t line = 1;
    std::size_t column = 1;
};

constexpr bool operator==(SourceLocation lhs, SourceLocation rhs) {
    return lhs.line == rhs.line && lhs.column == rhs.column;
}

constexpr bool operator!=(SourceLocation lhs, SourceLocation rhs) {
    return !(lhs == rhs);
}

} // namespace forgecc::common
