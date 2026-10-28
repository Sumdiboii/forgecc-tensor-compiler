#include "forgecc/frontend/lexer.hpp"

#include <cctype>
#include <string>

namespace forgecc::frontend {

Lexer::Lexer(std::string_view source) : source_(source) {}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;

    while (true) {
        skipWhitespace();

        if (atEnd()) {
            tokens.push_back(Token{TokenKind::EndOfFile, {}, location_});
            return tokens;
        }

        const char current = peek();
        if (std::isalpha(static_cast<unsigned char>(current)) || current == '_') {
            tokens.push_back(lexIdentifier());
        } else if (std::isdigit(static_cast<unsigned char>(current))) {
            tokens.push_back(lexNumber());
        } else {
            tokens.push_back(lexPunctuationOrOperator());
        }
    }
}

const common::DiagnosticEngine& Lexer::diagnostics() const {
    return diagnostics_;
}

bool Lexer::atEnd() const {
    return offset_ >= source_.size();
}

char Lexer::peek() const {
    return atEnd() ? '\0' : source_[offset_];
}

char Lexer::advance() {
    const char current = source_[offset_++];
    if (current == '\n') {
        ++location_.line;
        location_.column = 1;
    } else {
        ++location_.column;
    }

    return current;
}

void Lexer::skipWhitespace() {
    while (!atEnd()) {
        const char current = peek();
        if (current != ' ' && current != '\t' && current != '\r' && current != '\n') {
            return;
        }

        advance();
    }
}

Token Lexer::lexIdentifier() {
    const std::size_t start = offset_;
    const common::SourceLocation location = location_;

    advance();
    while (!atEnd()) {
        const char current = peek();
        if (!std::isalnum(static_cast<unsigned char>(current)) && current != '_') {
            break;
        }

        advance();
    }

    return makeToken(TokenKind::Identifier, start, location);
}

Token Lexer::lexNumber() {
    const std::size_t start = offset_;
    const common::SourceLocation location = location_;
    bool isFloating = false;

    while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    if (!atEnd() && peek() == '.') {
        isFloating = true;
        advance();
        while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    return makeToken(isFloating ? TokenKind::FloatingLiteral : TokenKind::IntegerLiteral,
                     start, location);
}

Token Lexer::makeToken(TokenKind kind, std::size_t start,
                       common::SourceLocation location) const {
    return Token{kind, std::string(source_.substr(start, offset_ - start)), location};
}

Token Lexer::lexPunctuationOrOperator() {
    const std::size_t start = offset_;
    const common::SourceLocation location = location_;
    const char current = advance();

    TokenKind kind = TokenKind::Invalid;
    switch (current) {
    case ':':
        kind = TokenKind::Colon;
        break;
    case ',':
        kind = TokenKind::Comma;
        break;
    case '[':
        kind = TokenKind::LeftBracket;
        break;
    case ']':
        kind = TokenKind::RightBracket;
        break;
    case '(':
        kind = TokenKind::LeftParenthesis;
        break;
    case ')':
        kind = TokenKind::RightParenthesis;
        break;
    case '=':
        kind = TokenKind::Equal;
        break;
    case '+':
        kind = TokenKind::Plus;
        break;
    case '-':
        kind = TokenKind::Minus;
        break;
    case '*':
        kind = TokenKind::Star;
        break;
    case '/':
        kind = TokenKind::Slash;
        break;
    default:
        diagnostics_.reportError(location, "invalid character in input");
        break;
    }

    return makeToken(kind, start, location);
}

} // namespace forgecc::frontend
