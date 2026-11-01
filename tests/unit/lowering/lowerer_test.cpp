#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"
#include "forgecc/frontend/semantic_analyzer.hpp"
#include "forgecc/lowering/lowerer.hpp"

#include <cassert>
#include <string_view>
#include <utility>

namespace {

using forgecc::frontend::Lexer;
using forgecc::frontend::Parser;
using forgecc::frontend::SemanticAnalyzer;
using forgecc::ir::OperationKind;
using forgecc::ir::ValueKind;
using forgecc::lowering::GraphLowerer;

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

std::optional<forgecc::ir::Graph> lowerProgram(std::string_view source,
                                               SemanticAnalyzer& analyzer,
                                               GraphLowerer& lowerer) {
    auto program = parseProgram(source);
    assert(analyzer.analyze(*program));
    return lowerer.lower(*program, analyzer);
}

void testDeclarationsBecomeInputs() {
    SemanticAnalyzer analyzer;
    GraphLowerer lowerer;
    auto graph = lowerProgram("A: f32[2, 3]\nB: f32[3, 4]", analyzer, lowerer);

    assert(graph.has_value());
    assert(graph->valueCount() == 2);
    assert(graph->nodeCount() == 0);
    assert(graph->value(0).kind == ValueKind::GraphInput);
    assert(graph->value(0).name == "A");
    assert(graph->value(0).type.dimensions == std::vector<std::size_t>({2, 3}));
    assert(graph->value(1).name == "B");
}

void testNestedCallsBecomeConnectedNodes() {
    SemanticAnalyzer analyzer;
    GraphLowerer lowerer;
    auto graph = lowerProgram(
        "A: f32[2, 3]\nB: f32[3, 4]\nC = relu(matmul(A, B))", analyzer, lowerer);

    assert(graph.has_value());
    assert(graph->nodeCount() == 2);
    const auto& matmul = graph->node(0);
    const auto& relu = graph->node(1);
    assert(matmul.operation == OperationKind::Matmul);
    assert(matmul.inputs == std::vector<forgecc::ir::ValueId>({0, 1}));
    assert(graph->value(matmul.output).type.dimensions ==
           std::vector<std::size_t>({2, 4}));
    assert(relu.operation == OperationKind::Relu);
    assert(relu.inputs == std::vector<forgecc::ir::ValueId>({matmul.output}));
    assert(graph->value(relu.output).name == "C");
    assert(graph->value(relu.output).producer == 1);
    assert(graph->value(relu.output).type.dimensions ==
           std::vector<std::size_t>({2, 4}));
}

void testSimpleReluAndMatmul() {
    for (const std::string_view source : {
             "A: f32[2, 2]\nC = relu(A)",
             "A: f32[2, 3]\nB: f32[3, 2]\nC = matmul(A, B)",
         }) {
        SemanticAnalyzer analyzer;
        GraphLowerer lowerer;
        assert(lowerProgram(source, analyzer, lowerer).has_value());
        assert(lowerer.diagnostics().diagnostics().empty());
    }
}

void testInvalidInputFailsSafely() {
    auto program = parseProgram("A: f32[2]\nC = relu(A)");
    SemanticAnalyzer analyzer;
    GraphLowerer lowerer;
    assert(analyzer.analyze(*program));

    auto& assignment = static_cast<forgecc::frontend::AssignmentStatement&>(*program->statements[1]);
    assignment.value = std::make_unique<forgecc::frontend::IdentifierExpression>(
        "missing", forgecc::common::SourceLocation{2, 5});
    assert(!lowerer.lower(*program, analyzer).has_value());
    assert(lowerer.diagnostics().hasErrors());
}

} // namespace

void runLowererTests() {
    testDeclarationsBecomeInputs();
    testNestedCallsBecomeConnectedNodes();
    testSimpleReluAndMatmul();
    testInvalidInputFailsSafely();
}
