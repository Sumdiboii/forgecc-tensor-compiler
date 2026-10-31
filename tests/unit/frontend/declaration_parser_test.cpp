#include "forgecc/frontend/ast.hpp"
#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"

#include <cassert>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using forgecc::common::SourceLocation;
using forgecc::frontend::Lexer;
using forgecc::frontend::Parser;
using forgecc::frontend::StatementKind;
using forgecc::frontend::TensorDeclarationStatement;

Parser makeParser(std::string_view source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    assert(!lexer.diagnostics().hasErrors());
    return Parser(std::move(tokens));
}

void testOneDimensionalDeclaration() {
    Parser parser = makeParser("bias: f32[256]");
    auto statement = parser.parseDeclaration();
    assert(statement);

    const auto& declaration = static_cast<const TensorDeclarationStatement&>(*statement);
    assert(declaration.name == "bias");
    assert(declaration.elementType == "f32");
    assert(declaration.dimensions.size() == 1);
    assert(declaration.dimensions[0] == 256);
    const SourceLocation location{1, 1};
    assert(declaration.location() == location);
}

void testMultidimensionalDeclaration() {
    Parser parser = makeParser("A: f32[2, 3, 4]");
    auto statement = parser.parseDeclaration();
    assert(statement);

    const auto& declaration = static_cast<const TensorDeclarationStatement&>(*statement);
    assert(declaration.dimensions == std::vector<std::size_t>({2, 3, 4}));
}

void testSupportedRanks() {
    for (const std::string_view source : {"A: f32[1]", "A: f32[1, 2]", "A: f32[1, 2, 3, 4]"}) {
        Parser parser = makeParser(source);
        auto statement = parser.parseDeclaration();
        assert(statement);
        const auto& declaration = static_cast<const TensorDeclarationStatement&>(*statement);
        assert(!declaration.dimensions.empty());
    }
}

void testMalformedDeclarations() {
    for (const std::string_view source : {
             ": f32[2]",
             "A f32[2]",
             "A:",
             "A: f64[2]",
             "A: f32 2]",
             "A: f32[",
             "A: f32[]",
             "A: f32[2,]",
             "A: f32[2,,3]",
             "A: f32[2 3]",
             "A: f32[2] extra",
             "A: f32[1.5]",
             "A: f32[-1]",
         }) {
        Parser parser = makeParser(source);
        assert(!parser.parseDeclaration());
        assert(parser.diagnostics().hasErrors());
        assert(!parser.diagnostics().diagnostics().empty());
    }

    const std::string tooLargeDimension =
        "A: f32[" + std::to_string(std::numeric_limits<std::size_t>::max()) + "0]";
    Parser overflowParser = makeParser(tooLargeDimension);
    assert(!overflowParser.parseDeclaration());
    assert(overflowParser.diagnostics().hasErrors());
}

} // namespace

void runDeclarationParserTests() {
    testOneDimensionalDeclaration();
    testMultidimensionalDeclaration();
    testSupportedRanks();
    testMalformedDeclarations();
}
