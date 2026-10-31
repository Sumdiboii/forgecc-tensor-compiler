#include "forgecc/frontend/ast.hpp"

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using forgecc::common::SourceLocation;
using forgecc::frontend::AssignmentStatement;
using forgecc::frontend::ExpressionKind;
using forgecc::frontend::IdentifierExpression;
using forgecc::frontend::Program;
using forgecc::frontend::StatementKind;
using forgecc::frontend::TensorDeclarationStatement;

void testTensorDeclaration() {
    const SourceLocation location{2, 1};
    TensorDeclarationStatement declaration("A", "f32", {2, 3, 4}, location);

    assert(declaration.kind() == StatementKind::TensorDeclaration);
    assert(declaration.name == "A");
    assert(declaration.elementType == "f32");
    assert(declaration.dimensions == std::vector<std::size_t>({2, 3, 4}));
    assert(declaration.location() == location);
}

void testDimensionShapes() {
    TensorDeclarationStatement vectorDeclaration("bias", "f32", {64}, {1, 1});
    TensorDeclarationStatement scalarDeclaration("value", "f32", {}, {1, 1});

    assert(vectorDeclaration.dimensions.size() == 1);
    assert(vectorDeclaration.dimensions[0] == 64);
    assert(scalarDeclaration.dimensions.empty());
}

void testEmptyProgram() {
    Program program;
    assert(program.statements.empty());
}

void testProgramStatementOrder() {
    std::vector<forgecc::frontend::StatementPtr> statements;
    statements.push_back(std::make_unique<TensorDeclarationStatement>(
        "A", "f32", std::vector<std::size_t>{2, 2}, SourceLocation{1, 1}));
    statements.push_back(std::make_unique<AssignmentStatement>(
        "out", std::make_unique<IdentifierExpression>("A", SourceLocation{2, 7}),
        SourceLocation{2, 1}));
    statements.push_back(std::make_unique<TensorDeclarationStatement>(
        "bias", "f32", std::vector<std::size_t>{2}, SourceLocation{3, 1}));

    Program program(std::move(statements));
    assert(program.statements.size() == 3);
    assert(program.statements[0]->kind() == StatementKind::TensorDeclaration);
    assert(program.statements[1]->kind() == StatementKind::Assignment);
    assert(program.statements[2]->kind() == StatementKind::TensorDeclaration);

    const auto& assignment = static_cast<const AssignmentStatement&>(*program.statements[1]);
    assert(assignment.target == "out");
    assert(assignment.value->kind() == ExpressionKind::Identifier);
}

} // namespace

void runProgramAstTests() {
    testTensorDeclaration();
    testDimensionShapes();
    testEmptyProgram();
    testProgramStatementOrder();
}
