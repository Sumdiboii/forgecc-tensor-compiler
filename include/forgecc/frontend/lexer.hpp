#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/frontend/token.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace forgecc::frontend {

class Lexer {
public:
    explicit Lexer(std::string_view source);

    [[nodiscard]] std::vector<Token> tokenize();
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    [[nodiscard]] bool atEnd() const;
    [[nodiscard]] char peek() const;
    char advance();
    void skipWhitespace();
    [[nodiscard]] Token lexIdentifier();
    [[nodiscard]] Token lexNumber();
    [[nodiscard]] Token makeToken(TokenKind kind, std::size_t start,
                                  common::SourceLocation location) const;
    [[nodiscard]] Token lexPunctuationOrOperator();

    std::string_view source_;
    std::size_t offset_ = 0;
    common::SourceLocation location_;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::frontend
