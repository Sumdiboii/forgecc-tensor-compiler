#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runCppEmitterTests();
void runLexerTests();
void runParserTests();
void runPart2ParserTests();
void runProgramAstTests();
void runDeclarationParserTests();
void runProgramParserTests();
void runTypeAndSymbolTableTests();
void runSemanticAnalyzerTests();
void runGraphTests();
void runLowererTests();
void runLoopIrTests();
void runLoopLowererTests();
void runPassManagerTests();
void runDeadCodeEliminationTests();
void runCommonSubexpressionEliminationTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runCppEmitterTests();
    runLexerTests();
    runParserTests();
    runPart2ParserTests();
    runProgramAstTests();
    runDeclarationParserTests();
    runProgramParserTests();
    runTypeAndSymbolTableTests();
    runSemanticAnalyzerTests();
    runGraphTests();
    runLowererTests();
    runLoopIrTests();
    runLoopLowererTests();
    runPassManagerTests();
    runDeadCodeEliminationTests();
    runCommonSubexpressionEliminationTests();
    std::cout << "ForgeCC unit tests passed\n";
}
