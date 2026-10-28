#include "forgecc/frontend/parser.hpp"

#include <utility>

namespace forgecc::frontend {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

ExpressionPtr Parser::parse() {
    ExpressionPtr expression = parsePrimary();
    if (!expression) {
        return nullptr;
    }

    if (!atEnd()) {
        reportError(currentToken().location, "unexpected token after expression");
        return nullptr;
    }

    return expression;
}

const common::DiagnosticEngine& Parser::diagnostics() const {
    return diagnostics_;
}

const Token& Parser::currentToken() const {
    static const Token endToken{TokenKind::EndOfFile, {}, {1, 1}};
    if (tokens_.empty() || current_ >= tokens_.size()) {
        return endToken;
    }

    return tokens_[current_];
}

bool Parser::atEnd() const {
    return currentToken().kind == TokenKind::EndOfFile;
}

const Token& Parser::advance() {
    const Token& token = currentToken();
    if (!atEnd() && current_ < tokens_.size()) {
        ++current_;
    }

    return token;
}

bool Parser::match(TokenKind kind) {
    if (currentToken().kind != kind) {
        return false;
    }

    advance();
    return true;
}

ExpressionPtr Parser::parsePrimary() {
    switch (currentToken().kind) {
    case TokenKind::Identifier:
        return parseIdentifierOrCall();
    case TokenKind::IntegerLiteral:
    case TokenKind::FloatingLiteral:
        return parseNumber();
    case TokenKind::LeftParenthesis:
        return parseParenthesized();
    case TokenKind::EndOfFile:
        reportError(currentToken().location, "expected expression, found end of input");
        return nullptr;
    case TokenKind::Invalid:
        reportError(currentToken().location, "cannot parse invalid token");
        return nullptr;
    default:
        reportError(currentToken().location, "expected expression");
        return nullptr;
    }
}

ExpressionPtr Parser::parseIdentifierOrCall() {
    const Token token = advance();
    if (currentToken().kind != TokenKind::LeftParenthesis) {
        return std::make_unique<IdentifierExpression>(token.lexeme, token.location);
    }

    return parseCall(token.lexeme, token.location);
}

ExpressionPtr Parser::parseCall(std::string callee, common::SourceLocation location) {
    advance();
    std::vector<ExpressionPtr> arguments;

    if (!match(TokenKind::RightParenthesis)) {
        while (true) {
            ExpressionPtr argument = parsePrimary();
            if (!argument) {
                return nullptr;
            }
            arguments.push_back(std::move(argument));

            if (match(TokenKind::RightParenthesis)) {
                break;
            }

            if (!match(TokenKind::Comma)) {
                reportError(currentToken().location, "expected ',' or ')' after argument");
                return nullptr;
            }
        }
    }

    return std::make_unique<CallExpression>(std::move(callee), std::move(arguments), location);
}

ExpressionPtr Parser::parseNumber() {
    const Token token = advance();
    const NumericLiteralKind literalKind = token.kind == TokenKind::IntegerLiteral
                                               ? NumericLiteralKind::Integer
                                               : NumericLiteralKind::Floating;
    return std::make_unique<NumericLiteralExpression>(literalKind, token.lexeme, token.location);
}

ExpressionPtr Parser::parseParenthesized() {
    const Token openingToken = advance();
    ExpressionPtr expression = parsePrimary();
    if (!expression) {
        return nullptr;
    }

    if (!match(TokenKind::RightParenthesis)) {
        reportError(currentToken().location, "expected ')' to close parenthesized expression");
        return nullptr;
    }

    return std::make_unique<ParenthesizedExpression>(std::move(expression), openingToken.location);
}

void Parser::reportError(common::SourceLocation location, std::string message) {
    diagnostics_.reportError(location, std::move(message));
}

} // namespace forgecc::frontend
