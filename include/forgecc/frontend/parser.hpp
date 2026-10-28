#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/frontend/ast.hpp"
#include "forgecc/frontend/token.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace forgecc::frontend {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    [[nodiscard]] ExpressionPtr parse();
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    [[nodiscard]] const Token& currentToken() const;
    [[nodiscard]] bool atEnd() const;
    const Token& advance();
    bool match(TokenKind kind);
    [[nodiscard]] ExpressionPtr parsePrimary();
    [[nodiscard]] ExpressionPtr parseIdentifierOrCall();
    [[nodiscard]] ExpressionPtr parseCall(std::string callee,
                                          common::SourceLocation location);
    [[nodiscard]] ExpressionPtr parseNumber();
    [[nodiscard]] ExpressionPtr parseParenthesized();
    void reportError(common::SourceLocation location, std::string message);

    std::vector<Token> tokens_;
    std::size_t current_ = 0;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::frontend
