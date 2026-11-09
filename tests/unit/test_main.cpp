#include <iostream>

void runCommandLineTests();
void runDiagnosticTests();
void runSourceLocationTests();
void runCppEmitterTests();
void runStandaloneEmitterTests();
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
void runMemoryPlanTests();
void runNativeCompilerTests();
void runEndToEndTests();
void runPassManagerTests();
void runDeadCodeEliminationTests();
void runCommonSubexpressionEliminationTests();

int main() {
    runCommandLineTests();
    runDiagnosticTests();
    runSourceLocationTests();
    runCppEmitterTests();
    runStandaloneEmitterTests();
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
    runMemoryPlanTests();
    runNativeCompilerTests();
    runEndToEndTests();
    runPassManagerTests();
    runDeadCodeEliminationTests();
    runCommonSubexpressionEliminationTests();
    std::cout << "ForgeCC unit tests passed\n";
}
