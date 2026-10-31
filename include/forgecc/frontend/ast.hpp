#pragma once

#include "forgecc/common/source_location.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace forgecc::frontend {

enum class ExpressionKind {
    Identifier,
    NumericLiteral,
    Parenthesized,
    Call,
    Binary,
};

class Expression {
public:
    virtual ~Expression() = default;

    [[nodiscard]] ExpressionKind kind() const {
        return kind_;
    }

    [[nodiscard]] common::SourceLocation location() const {
        return location_;
    }

protected:
    Expression(ExpressionKind kind, common::SourceLocation location)
        : kind_(kind), location_(location) {}

private:
    ExpressionKind kind_;
    common::SourceLocation location_;
};

using ExpressionPtr = std::unique_ptr<Expression>;

class IdentifierExpression final : public Expression {
public:
    IdentifierExpression(std::string name, common::SourceLocation location)
        : Expression(ExpressionKind::Identifier, location), name(std::move(name)) {}

    std::string name;
};

enum class NumericLiteralKind {
    Integer,
    Floating,
};

class NumericLiteralExpression final : public Expression {
public:
    NumericLiteralExpression(NumericLiteralKind literalKind, std::string lexeme,
                             common::SourceLocation location)
        : Expression(ExpressionKind::NumericLiteral, location), literalKind(literalKind),
          lexeme(std::move(lexeme)) {}

    NumericLiteralKind literalKind;
    std::string lexeme;
};

class ParenthesizedExpression final : public Expression {
public:
    ParenthesizedExpression(ExpressionPtr expression, common::SourceLocation location)
        : Expression(ExpressionKind::Parenthesized, location), expression(std::move(expression)) {}

    ExpressionPtr expression;
};

class CallExpression final : public Expression {
public:
    CallExpression(std::string callee, std::vector<ExpressionPtr> arguments,
                   common::SourceLocation location)
        : Expression(ExpressionKind::Call, location), callee(std::move(callee)),
          arguments(std::move(arguments)) {}

    std::string callee;
    std::vector<ExpressionPtr> arguments;
};

enum class BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
};

class BinaryExpression final : public Expression {
public:
    BinaryExpression(BinaryOperator operation, ExpressionPtr left, ExpressionPtr right,
                     common::SourceLocation location)
        : Expression(ExpressionKind::Binary, location), operation(operation),
          left(std::move(left)), right(std::move(right)) {}

    BinaryOperator operation;
    ExpressionPtr left;
    ExpressionPtr right;
};

enum class StatementKind {
    Assignment,
};

class Statement {
public:
    virtual ~Statement() = default;

    [[nodiscard]] StatementKind kind() const {
        return kind_;
    }

    [[nodiscard]] common::SourceLocation location() const {
        return location_;
    }

protected:
    Statement(StatementKind kind, common::SourceLocation location)
        : kind_(kind), location_(location) {}

private:
    StatementKind kind_;
    common::SourceLocation location_;
};

using StatementPtr = std::unique_ptr<Statement>;

class AssignmentStatement final : public Statement {
public:
    AssignmentStatement(std::string target, ExpressionPtr value,
                        common::SourceLocation location)
        : Statement(StatementKind::Assignment, location), target(std::move(target)),
          value(std::move(value)) {}

    std::string target;
    ExpressionPtr value;
};

} // namespace forgecc::frontend
