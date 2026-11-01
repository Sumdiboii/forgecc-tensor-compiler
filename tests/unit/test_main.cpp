#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runLexerTests();
void runParserTests();
void runPart2ParserTests();
void runProgramAstTests();
void runDeclarationParserTests();
void runProgramParserTests();
void runTypeAndSymbolTableTests();
void runSemanticAnalyzerTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runLexerTests();
    runParserTests();
    runPart2ParserTests();
    runProgramAstTests();
    runDeclarationParserTests();
    runProgramParserTests();
    runTypeAndSymbolTableTests();
    runSemanticAnalyzerTests();
    std::cout << "ForgeCC unit tests passed\n";
}
