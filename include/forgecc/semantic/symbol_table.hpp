#pragma once

#include "forgecc/semantic/type.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace forgecc::semantic {

struct Symbol {
    std::string name;
    TensorType type;
};

class SymbolTable {
public:
    // A duplicate insertion fails and leaves the existing symbol unchanged.
    [[nodiscard]] bool insert(Symbol symbol);
    [[nodiscard]] const Symbol* lookup(std::string_view name) const;
    [[nodiscard]] std::size_t size() const;

private:
    std::unordered_map<std::string, Symbol> symbols_;
};

} // namespace forgecc::semantic
