#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"
#include "forgecc/frontend/semantic_analyzer.hpp"

#include <cassert>
#include <string_view>
#include <utility>

namespace {

using forgecc::common::SourceLocation;
using forgecc::frontend::Lexer;
using forgecc::frontend::Parser;
using forgecc::frontend::SemanticAnalyzer;
using forgecc::frontend::StatementKind;

std::unique_ptr<forgecc::frontend::Program> parseProgram(std::string_view source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    assert(!lexer.diagnostics().hasErrors());

    Parser parser(std::move(tokens));
    auto program = parser.parseProgram();
    assert(program);
    assert(!parser.diagnostics().hasErrors());
    return program;
}

void testValidProgramAndSymbols() {
    auto program = parseProgram("A: f32[2, 2]\nB: f32[2, 2]\nout = relu(matmul(A, B))");
    SemanticAnalyzer analyzer;

    assert(analyzer.analyze(*program));
    assert(!analyzer.diagnostics().hasErrors());
    assert(analyzer.symbols().size() == 2);
    assert(analyzer.symbols().lookup("A") != nullptr);
    assert(analyzer.symbols().lookup("B") != nullptr);
    assert(analyzer.symbols().lookup("out") == nullptr);
}

void testUndeclaredIdentifiers() {
    for (const std::string_view source : {
             "out = missing",
             "out = A + (B * missing)",
             "out = relu(matmul(A, missing))",
         }) {
        auto program = parseProgram(source);
        SemanticAnalyzer analyzer;

        assert(!analyzer.analyze(*program));
        assert(analyzer.diagnostics().hasErrors());
        assert(!analyzer.diagnostics().diagnostics().empty());
        assert(analyzer.diagnostics().diagnostics().front().message.find("undeclared identifier") !=
               std::string::npos);
    }
}

void testDuplicateDeclarations() {
    auto program = parseProgram("A: f32[2]\nA: f32[3]");
    SemanticAnalyzer analyzer;

    assert(!analyzer.analyze(*program));
    assert(analyzer.diagnostics().diagnostics().size() == 1);
    const auto& diagnostic = analyzer.diagnostics().diagnostics().front();
    assert(diagnostic.message.find("duplicate tensor declaration") != std::string::npos);
    const SourceLocation expectedLocation{2, 1};
    assert(diagnostic.location == expectedLocation);
    assert(analyzer.symbols().lookup("A")->type.dimensions[0] == 2);
}

void testOperationValidation() {
    for (const std::string_view source : {
             "out = custom_function(A)",
             "out = relu(A, B)",
             "out = matmul(A)",
         }) {
        auto program = parseProgram(source);
        SemanticAnalyzer analyzer;

        assert(!analyzer.analyze(*program));
        assert(analyzer.diagnostics().hasErrors());
    }
}

void testAssignmentsDoNotRequireDeclaredTargets() {
    auto program = parseProgram("A: f32[2]\nout = A");
    SemanticAnalyzer analyzer;

    assert(analyzer.analyze(*program));
    assert(analyzer.symbols().lookup("out") == nullptr);
}

void testRegressionStatementKinds() {
    auto program = parseProgram("A: f32[2]\nout = relu(A)");
    assert(program->statements[0]->kind() == StatementKind::TensorDeclaration);
    assert(program->statements[1]->kind() == StatementKind::Assignment);
}

} // namespace

void runSemanticAnalyzerTests() {
    testValidProgramAndSymbols();
    testUndeclaredIdentifiers();
    testDuplicateDeclarations();
    testOperationValidation();
    testAssignmentsDoNotRequireDeclaredTargets();
    testRegressionStatementKinds();
}
