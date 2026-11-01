#include "forgecc/lowering/lowerer.hpp"

#include <utility>

namespace forgecc::lowering {

std::optional<ir::Graph> GraphLowerer::lower(const frontend::Program& program,
                                             frontend::SemanticAnalyzer& analyzer) {
    analyzer_ = &analyzer;
    graph_ = ir::Graph{};
    values_.clear();
    diagnostics_ = common::DiagnosticEngine{};

    if (analyzer.diagnostics().hasErrors()) {
        reportError({1, 1}, "cannot lower a program with semantic errors");
        return std::nullopt;
    }

    for (const frontend::StatementPtr& statement : program.statements) {
        if (statement->kind() == frontend::StatementKind::TensorDeclaration) {
            const auto& declaration =
                static_cast<const frontend::TensorDeclarationStatement&>(*statement);
            const semantic::Symbol* symbol = analyzer.symbols().lookup(declaration.name);
            if (symbol == nullptr) {
                reportError(declaration.location(),
                            "tensor declaration is missing from semantic symbols");
                return std::nullopt;
            }

            const ir::ValueId value = graph_.addInput(symbol->name, symbol->type);
            if (!values_.emplace(symbol->name, value).second) {
                reportError(declaration.location(),
                            "duplicate graph value for tensor '" + symbol->name + "'");
                return std::nullopt;
            }
            continue;
        }

        const auto& assignment =
            static_cast<const frontend::AssignmentStatement&>(*statement);
        const std::optional<ir::ValueId> value = lowerExpression(*assignment.value);
        if (!value.has_value()) {
            return std::nullopt;
        }

        values_[assignment.target] = *value;
        if (graph_.value(*value).kind == ir::ValueKind::OperationOutput) {
            graph_.setValueName(*value, assignment.target);
        }
    }

    return std::move(graph_);
}

const common::DiagnosticEngine& GraphLowerer::diagnostics() const {
    return diagnostics_;
}

std::optional<ir::ValueId> GraphLowerer::lowerExpression(
    const frontend::Expression& expression) {
    switch (expression.kind()) {
    case frontend::ExpressionKind::Identifier: {
        const auto& identifier = static_cast<const frontend::IdentifierExpression&>(expression);
        const auto value = values_.find(identifier.name);
        if (value == values_.end()) {
            reportError(identifier.location(), "no graph value for identifier '" +
                                                    identifier.name + "'");
            return std::nullopt;
        }
        return value->second;
    }
    case frontend::ExpressionKind::Parenthesized: {
        const auto& parenthesized =
            static_cast<const frontend::ParenthesizedExpression&>(expression);
        return lowerExpression(*parenthesized.expression);
    }
    case frontend::ExpressionKind::Call:
        return lowerCall(static_cast<const frontend::CallExpression&>(expression));
    case frontend::ExpressionKind::NumericLiteral:
        reportError(expression.location(), "numeric literals cannot be lowered to the tensor graph");
        return std::nullopt;
    case frontend::ExpressionKind::Binary:
        reportError(expression.location(), "binary expressions are not supported by the graph IR");
        return std::nullopt;
    }

    return std::nullopt;
}

std::optional<ir::ValueId> GraphLowerer::lowerCall(const frontend::CallExpression& call) {
    ir::OperationKind operation;
    if (call.callee == "relu") {
        if (call.arguments.size() != 1) {
            reportError(call.location(), "relu expects exactly one argument");
            return std::nullopt;
        }
        operation = ir::OperationKind::Relu;
    } else if (call.callee == "matmul") {
        if (call.arguments.size() != 2) {
            reportError(call.location(), "matmul expects exactly two arguments");
            return std::nullopt;
        }
        operation = ir::OperationKind::Matmul;
    } else {
        reportError(call.location(), "unsupported function '" + call.callee + "'");
        return std::nullopt;
    }

    std::vector<ir::ValueId> inputs;
    inputs.reserve(call.arguments.size());
    for (const frontend::ExpressionPtr& argument : call.arguments) {
        const std::optional<ir::ValueId> input = lowerExpression(*argument);
        if (!input.has_value()) {
            return std::nullopt;
        }
        inputs.push_back(*input);
    }

    const std::optional<semantic::TensorType> outputType = analyzer_->inferShape(call);
    if (!outputType.has_value()) {
        reportError(call.location(), "unable to infer graph operation output shape");
        return std::nullopt;
    }

    const ir::NodeId node = graph_.addNode(operation, std::move(inputs), *outputType);
    return graph_.node(node).output;
}

void GraphLowerer::reportError(common::SourceLocation location, std::string message) {
    diagnostics_.reportError(location, std::move(message));
}

} // namespace forgecc::lowering
