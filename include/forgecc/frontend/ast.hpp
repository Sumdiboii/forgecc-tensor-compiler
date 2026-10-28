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

} // namespace forgecc::frontend
