#include "forgecc/semantic/symbol_table.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

namespace {

using forgecc::semantic::ElementType;
using forgecc::semantic::Symbol;
using forgecc::semantic::SymbolTable;
using forgecc::semantic::TensorType;

void testTensorType() {
    TensorType type(ElementType::F32, {2, 3, 4});

    assert(type.elementType == ElementType::F32);
    assert(type.dimensions == std::vector<std::size_t>({2, 3, 4}));
    assert(type.dimensions[0] == 2);
    assert(type.dimensions[1] == 3);
    assert(type.dimensions[2] == 4);
}

void testSymbolInsertionAndLookup() {
    SymbolTable table;
    const bool inserted = table.insert(Symbol{"A", TensorType(ElementType::F32, {2, 2})});

    assert(inserted);
    assert(table.size() == 1);

    const Symbol* symbol = table.lookup("A");
    assert(symbol != nullptr);
    assert(symbol->name == "A");
    assert(symbol->type.elementType == ElementType::F32);
    assert(symbol->type.dimensions == std::vector<std::size_t>({2, 2}));
}

void testMissingAndDuplicateSymbols() {
    SymbolTable table;
    assert(table.lookup("missing") == nullptr);

    assert(table.insert(Symbol{"A", TensorType(ElementType::F32, {2})}));
    assert(!table.insert(Symbol{"A", TensorType(ElementType::F32, {8})}));
    assert(table.size() == 1);

    const Symbol* symbol = table.lookup("A");
    assert(symbol != nullptr);
    assert(symbol->type.dimensions == std::vector<std::size_t>({2}));
}

} // namespace

void runTypeAndSymbolTableTests() {
    testTensorType();
    testSymbolInsertionAndLookup();
    testMissingAndDuplicateSymbols();
}
