#include "forgecc/frontend/ast.hpp"
#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"

#include <cassert>
#include <string_view>
#include <utility>

namespace {

using forgecc::frontend::AssignmentStatement;
using forgecc::frontend::BinaryExpression;
using forgecc::frontend::ExpressionKind;
using forgecc::frontend::Lexer;
using forgecc::frontend::Parser;
using forgecc::frontend::Program;
using forgecc::frontend::StatementKind;
using forgecc::frontend::TensorDeclarationStatement;

Parser makeParser(std::string_view source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    assert(!lexer.diagnostics().hasErrors());
    return Parser(std::move(tokens));
}

void testEmptyProgram() {
    Parser parser = makeParser("");
    auto program = parser.parseProgram();
    assert(program);
    assert(program->statements.empty());
}

void testSingleStatements() {
    Parser declarationParser = makeParser("A: f32[2, 2]");
    auto declarationProgram = declarationParser.parseProgram();
    assert(declarationProgram);
    assert(declarationProgram->statements.size() == 1);
    assert(declarationProgram->statements[0]->kind() == StatementKind::TensorDeclaration);

    Parser assignmentParser = makeParser("out = A + B");
    auto assignmentProgram = assignmentParser.parseProgram();
    assert(assignmentProgram);
    assert(assignmentProgram->statements.size() == 1);
    const auto& assignment =
        static_cast<const AssignmentStatement&>(*assignmentProgram->statements[0]);
    assert(assignment.target == "out");
    assert(assignment.value->kind() == ExpressionKind::Binary);
}

void testStatementOrder() {
    Parser parser = makeParser("A: f32[2, 2]\nB: f32[2, 2]\nout = A + B\nbias = out");
    auto program = parser.parseProgram();
    assert(program);
    assert(program->statements.size() == 4);
    assert(program->statements[0]->kind() == StatementKind::TensorDeclaration);
    assert(program->statements[1]->kind() == StatementKind::TensorDeclaration);
    assert(program->statements[2]->kind() == StatementKind::Assignment);
    assert(program->statements[3]->kind() == StatementKind::Assignment);

    const auto& firstDeclaration =
        static_cast<const TensorDeclarationStatement&>(*program->statements[0]);
    const auto& secondDeclaration =
        static_cast<const TensorDeclarationStatement&>(*program->statements[1]);
    const auto& firstAssignment =
        static_cast<const AssignmentStatement&>(*program->statements[2]);
    assert(firstDeclaration.name == "A");
    assert(secondDeclaration.name == "B");
    assert(firstAssignment.target == "out");
    assert(firstAssignment.value->kind() == ExpressionKind::Binary);
}

void testMultilineFunctionExpression() {
    Parser parser = makeParser("A: f32[2, 2]\nB: f32[2, 2]\nout = relu(matmul(A, B) + A)");
    auto program = parser.parseProgram();
    assert(program);
    assert(program->statements.size() == 3);

    const auto& assignment =
        static_cast<const AssignmentStatement&>(*program->statements[2]);
    assert(assignment.value->kind() == ExpressionKind::Call);
    const auto& call = static_cast<const forgecc::frontend::CallExpression&>(*assignment.value);
    assert(call.arguments.size() == 1);
    assert(call.arguments[0]->kind() == ExpressionKind::Binary);
    const auto& addition = static_cast<const BinaryExpression&>(*call.arguments[0]);
    assert(addition.left->kind() == ExpressionKind::Call);
    assert(addition.right->kind() == ExpressionKind::Identifier);
}

void testMalformedPrograms() {
    for (const std::string_view source : {
             "A: f32[2]\n= B",
             "A: f32[2]\nout =",
             "A: f32[2] trailing",
             "out = A B",
         }) {
        Parser parser = makeParser(source);
        assert(!parser.parseProgram());
        assert(parser.diagnostics().hasErrors());
    }
}

void testAmbiguousSameLineStatementsAreRejected() {
    Parser parser = makeParser("A: f32[2] B: f32[3]");
    assert(!parser.parseProgram());
    assert(parser.diagnostics().hasErrors());
}

} // namespace

void runProgramParserTests() {
    testEmptyProgram();
    testSingleStatements();
    testStatementOrder();
    testMultilineFunctionExpression();
    testMalformedPrograms();
    testAmbiguousSameLineStatementsAreRejected();
}
