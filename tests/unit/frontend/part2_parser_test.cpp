#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"

#include <cassert>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using forgecc::frontend::AssignmentStatement;
using forgecc::frontend::BinaryExpression;
using forgecc::frontend::BinaryOperator;
using forgecc::frontend::ExpressionKind;
using forgecc::frontend::Lexer;
using forgecc::frontend::Parser;
using forgecc::frontend::StatementKind;
using forgecc::common::SourceLocation;

Parser makeParser(std::string_view source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    assert(!lexer.diagnostics().hasErrors());
    return Parser(std::move(tokens));
}

void testPrecedenceAndAssociativity() {
    Parser parser = makeParser("A + B * C - D / E");
    auto expression = parser.parse();
    assert(expression);

    const auto& subtraction = static_cast<const BinaryExpression&>(*expression);
    assert(subtraction.operation == BinaryOperator::Subtract);
    assert(subtraction.left->kind() == ExpressionKind::Binary);
    assert(subtraction.right->kind() == ExpressionKind::Binary);

    const auto& addition = static_cast<const BinaryExpression&>(*subtraction.left);
    assert(addition.operation == BinaryOperator::Add);
    const auto& multiplication = static_cast<const BinaryExpression&>(*addition.right);
    assert(multiplication.operation == BinaryOperator::Multiply);
    const auto& division = static_cast<const BinaryExpression&>(*subtraction.right);
    assert(division.operation == BinaryOperator::Divide);

    Parser associativeParser = makeParser("A - B - C");
    auto associativeExpression = associativeParser.parse();
    assert(associativeExpression);
    const auto& outerSubtraction = static_cast<const BinaryExpression&>(*associativeExpression);
    assert(outerSubtraction.operation == BinaryOperator::Subtract);
    assert(outerSubtraction.left->kind() == ExpressionKind::Binary);
    assert(outerSubtraction.right->kind() == ExpressionKind::Identifier);
}

void testParenthesesAndFunctionArguments() {
    Parser parser = makeParser("relu(A * (B + C))");
    auto expression = parser.parse();
    assert(expression);
    assert(expression->kind() == ExpressionKind::Call);

    Parser argumentsParser = makeParser("custom(A + B, C * D)");
    auto arguments = argumentsParser.parse();
    assert(arguments);
    const auto& call = static_cast<const forgecc::frontend::CallExpression&>(*arguments);
    assert(call.arguments.size() == 2);
    assert(call.arguments[0]->kind() == ExpressionKind::Binary);
    assert(call.arguments[1]->kind() == ExpressionKind::Binary);
}

void testAssignment() {
    Parser parser = makeParser("out = A + B * bias");
    auto statement = parser.parseStatement();
    assert(statement);
    assert(statement->kind() == StatementKind::Assignment);

    const auto& assignment = static_cast<const AssignmentStatement&>(*statement);
    assert(assignment.target == "out");
    assert(assignment.value->kind() == ExpressionKind::Binary);
    const SourceLocation assignmentLocation{1, 1};
    assert(assignment.location() == assignmentLocation);
}

void testMalformedExpressionsAndAssignments() {
    for (const std::string_view source : {"A +", "A *", "(A + B", "A + * B"}) {
        Parser parser = makeParser(source);
        assert(!parser.parse());
        assert(parser.diagnostics().hasErrors());
    }

    for (const std::string_view source : {"out =", "out = A +", "out = A B"}) {
        Parser parser = makeParser(source);
        assert(!parser.parseStatement());
        assert(parser.diagnostics().hasErrors());
    }
}

} // namespace

void runPart2ParserTests() {
    testPrecedenceAndAssociativity();
    testParenthesesAndFunctionArguments();
    testAssignment();
    testMalformedExpressionsAndAssignments();
}
