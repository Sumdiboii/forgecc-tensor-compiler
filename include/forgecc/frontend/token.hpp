#pragma once

#include "forgecc/common/source_location.hpp"

#include <string>

namespace forgecc::frontend {

enum class TokenKind {
    Identifier,
    IntegerLiteral,
    FloatingLiteral,
    Colon,
    Comma,
    LeftBracket,
    RightBracket,
    LeftParenthesis,
    RightParenthesis,
    Equal,
    Plus,
    Minus,
    Star,
    Slash,
    EndOfFile,
    Invalid,
};

struct Token {
    TokenKind kind;
    std::string lexeme;
    common::SourceLocation location;
};

} // namespace forgecc::frontend
