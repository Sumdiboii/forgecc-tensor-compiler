#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runLexerTests();
void runParserTests();
void runPart2ParserTests();
void runProgramAstTests();
void runDeclarationParserTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runLexerTests();
    runParserTests();
    runPart2ParserTests();
    runProgramAstTests();
    runDeclarationParserTests();
    std::cout << "ForgeCC unit tests passed\n";
}
