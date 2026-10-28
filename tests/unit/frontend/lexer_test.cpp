#include "forgecc/frontend/lexer.hpp"

#include <cassert>
#include <string_view>

namespace {

using forgecc::common::SourceLocation;
using forgecc::frontend::Lexer;
using forgecc::frontend::TokenKind;

void assertToken(const forgecc::frontend::Token& token, TokenKind kind, std::string_view lexeme,
                 SourceLocation location) {
    assert(token.kind == kind);
    assert(token.lexeme == lexeme);
    assert(token.location == location);
}

void testEmptyInput() {
    const auto tokens = Lexer("").tokenize();
    assert(tokens.size() == 1);
    assertToken(tokens[0], TokenKind::EndOfFile, "", {1, 1});
}

void testIdentifiersAndNumbers() {
    const auto tokens = Lexer("A bias 128 256.5").tokenize();
    assert(tokens.size() == 5);
    assertToken(tokens[0], TokenKind::Identifier, "A", {1, 1});
    assertToken(tokens[1], TokenKind::Identifier, "bias", {1, 3});
    assertToken(tokens[2], TokenKind::IntegerLiteral, "128", {1, 8});
    assertToken(tokens[3], TokenKind::FloatingLiteral, "256.5", {1, 12});
    assertToken(tokens[4], TokenKind::EndOfFile, "", {1, 17});
}

void testPunctuationAndOperators() {
    const auto tokens = Lexer(":,[]()=+-*/").tokenize();
    const TokenKind expected[] = {
        TokenKind::Colon,          TokenKind::Comma,         TokenKind::LeftBracket,
        TokenKind::RightBracket,   TokenKind::LeftParenthesis, TokenKind::RightParenthesis,
        TokenKind::Equal,          TokenKind::Plus,           TokenKind::Minus,
        TokenKind::Star,           TokenKind::Slash,          TokenKind::EndOfFile,
    };

    assert(tokens.size() == sizeof(expected) / sizeof(expected[0]));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        assert(tokens[index].kind == expected[index]);
    }
}

void testTensorDeclaration() {
    const auto tokens = Lexer("A: f32[128, 256]").tokenize();
    const TokenKind expected[] = {
        TokenKind::Identifier, TokenKind::Colon,        TokenKind::Identifier,
        TokenKind::LeftBracket, TokenKind::IntegerLiteral, TokenKind::Comma,
        TokenKind::IntegerLiteral, TokenKind::RightBracket, TokenKind::EndOfFile,
    };

    assert(tokens.size() == sizeof(expected) / sizeof(expected[0]));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        assert(tokens[index].kind == expected[index]);
    }
    assert(tokens[2].lexeme == "f32");
}

void testLocationsAcrossLines() {
    const auto tokens = Lexer("A\n  +\nB").tokenize();
    assertToken(tokens[0], TokenKind::Identifier, "A", {1, 1});
    assertToken(tokens[1], TokenKind::Plus, "+", {2, 3});
    assertToken(tokens[2], TokenKind::Identifier, "B", {3, 1});
    assertToken(tokens[3], TokenKind::EndOfFile, "", {3, 2});
}

void testInvalidCharacter() {
    Lexer lexer("A @ B");
    const auto tokens = lexer.tokenize();

    assert(tokens.size() == 4);
    assert(tokens[1].kind == TokenKind::Invalid);
    assert(tokens[1].lexeme == "@");
    assert(tokens[1].location == SourceLocation(1, 3));
    assert(lexer.diagnostics().hasErrors());
    assert(lexer.diagnostics().diagnostics().front().location == SourceLocation(1, 3));
}

} // namespace

void runLexerTests() {
    testEmptyInput();
    testIdentifiersAndNumbers();
    testPunctuationAndOperators();
    testTensorDeclaration();
    testLocationsAcrossLines();
    testInvalidCharacter();
}
