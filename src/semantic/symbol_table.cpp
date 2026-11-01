#include "forgecc/semantic/symbol_table.hpp"

#include <string>
#include <utility>

namespace forgecc::semantic {

bool SymbolTable::insert(Symbol symbol) {
    if (symbols_.find(symbol.name) != symbols_.end()) {
        return false;
    }

    symbols_.emplace(symbol.name, std::move(symbol));
    return true;
}

const Symbol* SymbolTable::lookup(std::string_view name) const {
    const auto iterator = symbols_.find(std::string(name));
    if (iterator == symbols_.end()) {
        return nullptr;
    }

    return &iterator->second;
}

std::size_t SymbolTable::size() const {
    return symbols_.size();
}

} // namespace forgecc::semantic
