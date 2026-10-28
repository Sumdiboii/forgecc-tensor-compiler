#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"

#include <cassert>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using forgecc::common::SourceLocation;
using forgecc::frontend::CallExpression;
using forgecc::frontend::ExpressionKind;
using forgecc::frontend::IdentifierExpression;
using forgecc::frontend::Lexer;
using forgecc::frontend::NumericLiteralExpression;
using forgecc::frontend::NumericLiteralKind;
using forgecc::frontend::ParenthesizedExpression;
using forgecc::frontend::Parser;

Parser makeParser(std::string_view source) {
    Lexer lexer(source);
    std::vector<forgecc::frontend::Token> tokens = lexer.tokenize();
    assert(!lexer.diagnostics().hasErrors());
    return Parser(std::move(tokens));
}

void testIdentifiersAndNumbers() {
    Parser identifierParser = makeParser("A");
    auto identifier = identifierParser.parse();
    assert(identifier);
    assert(identifier->kind() == ExpressionKind::Identifier);
    assert(static_cast<IdentifierExpression&>(*identifier).name == "A");

    Parser integerParser = makeParser("42");
    auto integer = integerParser.parse();
    assert(integer);
    const auto& integerLiteral = static_cast<const NumericLiteralExpression&>(*integer);
    assert(integerLiteral.literalKind == NumericLiteralKind::Integer);
    assert(integerLiteral.lexeme == "42");

    Parser floatingParser = makeParser("256.5");
    auto floating = floatingParser.parse();
    assert(floating);
    const auto& floatingLiteral = static_cast<const NumericLiteralExpression&>(*floating);
    assert(floatingLiteral.literalKind == NumericLiteralKind::Floating);
    assert(floatingLiteral.lexeme == "256.5");
}

void testFunctionCalls() {
    Parser parser = makeParser("custom_function(A, B, 42)");
    auto expression = parser.parse();
    assert(expression);
    const auto& call = static_cast<const CallExpression&>(*expression);
    assert(call.callee == "custom_function");
    assert(call.arguments.size() == 3);
    assert(call.arguments[0]->kind() == ExpressionKind::Identifier);
    assert(call.arguments[1]->kind() == ExpressionKind::Identifier);
    assert(call.arguments[2]->kind() == ExpressionKind::NumericLiteral);

    Parser emptyCallParser = makeParser("relu()");
    auto emptyCall = emptyCallParser.parse();
    assert(emptyCall);
    assert(static_cast<const CallExpression&>(*emptyCall).arguments.empty());
}

void testNestedAndParenthesizedExpressions() {
    Parser nestedParser = makeParser("relu(matmul(A, B))");
    auto nested = nestedParser.parse();
    assert(nested);
    const auto& outerCall = static_cast<const CallExpression&>(*nested);
    assert(outerCall.callee == "relu");
    assert(outerCall.arguments.size() == 1);
    const auto& innerCall = static_cast<const CallExpression&>(*outerCall.arguments[0]);
    assert(innerCall.callee == "matmul");
    assert(innerCall.arguments.size() == 2);

    Parser parenthesizedParser = makeParser("(A)");
    auto parenthesized = parenthesizedParser.parse();
    assert(parenthesized);
    const auto& grouped = static_cast<const ParenthesizedExpression&>(*parenthesized);
    assert(grouped.expression->kind() == ExpressionKind::Identifier);
}

void testLocations() {
    Parser parser = makeParser("\n  relu(A)");
    auto expression = parser.parse();
    assert(expression);
    assert(expression->location() == SourceLocation(2, 3));
    const auto& call = static_cast<const CallExpression&>(*expression);
    assert(call.arguments[0]->location() == SourceLocation(2, 8));
}

void testParseErrors() {
    for (const std::string_view source : {"", "relu(", "relu(A", "relu(, A)", "relu(A,)",
                                           "relu(A B)"}) {
        Parser parser = makeParser(source);
        auto expression = parser.parse();
        assert(!expression);
        assert(parser.diagnostics().hasErrors());
        assert(!parser.diagnostics().diagnostics().empty());
    }

    Parser unmatchedParser = makeParser("(A");
    assert(!unmatchedParser.parse());
    assert(unmatchedParser.diagnostics().diagnostics().front().location == SourceLocation(1, 3));
}

} // namespace

void runParserTests() {
    testIdentifiersAndNumbers();
    testFunctionCalls();
    testNestedAndParenthesizedExpressions();
    testLocations();
    testParseErrors();
}
