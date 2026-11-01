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

std::optional<semantic::TensorType> SemanticAnalyzer::inferShape(const Expression& expression) {
    return inferExpression(expression);
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
    static_cast<void>(inferExpression(*assignment.value));
}

std::optional<semantic::TensorType> SemanticAnalyzer::inferExpression(const Expression& expression) {
    switch (expression.kind()) {
    case ExpressionKind::Identifier: {
        const auto& identifier = static_cast<const IdentifierExpression&>(expression);
        if (symbols_.lookup(identifier.name) == nullptr) {
            reportError(identifier.location(), "undeclared identifier '" + identifier.name + "'");
            return std::nullopt;
        }

        return symbols_.lookup(identifier.name)->type;
    }
    case ExpressionKind::NumericLiteral:
        return std::nullopt;
    case ExpressionKind::Parenthesized: {
        const auto& parenthesized = static_cast<const ParenthesizedExpression&>(expression);
        return inferExpression(*parenthesized.expression);
    }
    case ExpressionKind::Call:
        return inferCall(static_cast<const CallExpression&>(expression));
    case ExpressionKind::Binary: {
        const auto& binary = static_cast<const BinaryExpression&>(expression);
        return inferBinary(binary);
    }
    }

    return std::nullopt;
}

std::optional<semantic::TensorType> SemanticAnalyzer::inferCall(const CallExpression& call) {
    std::vector<std::optional<semantic::TensorType>> argumentShapes;
    argumentShapes.reserve(call.arguments.size());
    for (const ExpressionPtr& argument : call.arguments) {
        argumentShapes.push_back(inferExpression(*argument));
    }

    if (call.callee == "relu") {
        if (call.arguments.size() != 1) {
            reportError(call.location(), "relu expects exactly one argument");
            return std::nullopt;
        }

        return argumentShapes[0];
    } else if (call.callee == "matmul") {
        if (call.arguments.size() != 2) {
            reportError(call.location(), "matmul expects exactly two arguments");
            return std::nullopt;
        }

        if (!argumentShapes[0].has_value() || !argumentShapes[1].has_value()) {
            return std::nullopt;
        }

        const semantic::TensorType& left = *argumentShapes[0];
        const semantic::TensorType& right = *argumentShapes[1];
        if (left.dimensions.size() != 2) {
            reportError(call.arguments[0]->location(), "matmul requires rank-2 operands");
        }
        if (right.dimensions.size() != 2) {
            reportError(call.arguments[1]->location(), "matmul requires rank-2 operands");
        }
        if (left.dimensions.size() != 2 || right.dimensions.size() != 2) {
            return std::nullopt;
        }

        if (left.dimensions[1] != right.dimensions[0]) {
            reportError(call.location(), "matmul inner dimensions are incompatible");
            return std::nullopt;
        }

        return semantic::TensorType(semantic::ElementType::F32,
                                    {left.dimensions[0], right.dimensions[1]});
    } else {
        reportError(call.location(), "unsupported function '" + call.callee + "'");
        return std::nullopt;
    }
}

std::optional<semantic::TensorType> SemanticAnalyzer::inferBinary(
    const BinaryExpression& binary) {
    const std::optional<semantic::TensorType> left = inferExpression(*binary.left);
    const std::optional<semantic::TensorType> right = inferExpression(*binary.right);
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }

    if (left->dimensions != right->dimensions) {
        reportError(binary.location(), "binary operands have incompatible tensor shapes");
        return std::nullopt;
    }

    return left;
}

void SemanticAnalyzer::reportError(common::SourceLocation location, std::string message) {
    diagnostics_.reportError(location, std::move(message));
}

} // namespace forgecc::frontend
