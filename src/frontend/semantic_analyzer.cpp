#include "forgecc/frontend/semantic_analyzer.hpp"

#include <string>
#include <utility>

namespace forgecc::frontend {

bool SemanticAnalyzer::analyze(const Program& program) {
    symbols_ = semantic::SymbolTable{};
    diagnostics_ = common::DiagnosticEngine{};

    for (const StatementPtr& statement : program.statements) {
        analyzeStatement(*statement);
    }

    return !diagnostics_.hasErrors();
}

const common::DiagnosticEngine& SemanticAnalyzer::diagnostics() const {
    return diagnostics_;
}

const semantic::SymbolTable& SemanticAnalyzer::symbols() const {
    return symbols_;
}

void SemanticAnalyzer::analyzeStatement(const Statement& statement) {
    switch (statement.kind()) {
    case StatementKind::TensorDeclaration:
        analyzeDeclaration(static_cast<const TensorDeclarationStatement&>(statement));
        break;
    case StatementKind::Assignment:
        analyzeAssignment(static_cast<const AssignmentStatement&>(statement));
        break;
    }
}

void SemanticAnalyzer::analyzeDeclaration(const TensorDeclarationStatement& declaration) {
    if (declaration.elementType != "f32") {
        reportError(declaration.location(), "unsupported tensor element type");
        return;
    }

    semantic::TensorType type(semantic::ElementType::F32, declaration.dimensions);
    if (!symbols_.insert(semantic::Symbol{declaration.name, std::move(type)})) {
        reportError(declaration.location(),
                    "duplicate tensor declaration '" + declaration.name + "'");
    }
}

void SemanticAnalyzer::analyzeAssignment(const AssignmentStatement& assignment) {
    // Assignment targets introduce result bindings; their inferred type is deferred.
    analyzeExpression(*assignment.value);
}

void SemanticAnalyzer::analyzeExpression(const Expression& expression) {
    switch (expression.kind()) {
    case ExpressionKind::Identifier: {
        const auto& identifier = static_cast<const IdentifierExpression&>(expression);
        if (symbols_.lookup(identifier.name) == nullptr) {
            reportError(identifier.location(), "undeclared identifier '" + identifier.name + "'");
        }
        break;
    }
    case ExpressionKind::NumericLiteral:
        break;
    case ExpressionKind::Parenthesized: {
        const auto& parenthesized = static_cast<const ParenthesizedExpression&>(expression);
        analyzeExpression(*parenthesized.expression);
        break;
    }
    case ExpressionKind::Call:
        analyzeCall(static_cast<const CallExpression&>(expression));
        break;
    case ExpressionKind::Binary: {
        const auto& binary = static_cast<const BinaryExpression&>(expression);
        analyzeExpression(*binary.left);
        analyzeExpression(*binary.right);
        break;
    }
    }
}

void SemanticAnalyzer::analyzeCall(const CallExpression& call) {
    if (call.callee == "relu") {
        if (call.arguments.size() != 1) {
            reportError(call.location(), "relu expects exactly one argument");
        }
    } else if (call.callee == "matmul") {
        if (call.arguments.size() != 2) {
            reportError(call.location(), "matmul expects exactly two arguments");
        }
    } else {
        reportError(call.location(), "unsupported function '" + call.callee + "'");
    }

    for (const ExpressionPtr& argument : call.arguments) {
        analyzeExpression(*argument);
    }
}

void SemanticAnalyzer::reportError(common::SourceLocation location, std::string message) {
    diagnostics_.reportError(location, std::move(message));
}

} // namespace forgecc::frontend
