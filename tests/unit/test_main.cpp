#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runLexerTests();
void runParserTests();
void runPart2ParserTests();
void runProgramAstTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runLexerTests();
    runParserTests();
    runPart2ParserTests();
    runProgramAstTests();
    std::cout << "ForgeCC unit tests passed\n";
}
