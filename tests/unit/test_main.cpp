#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runLexerTests();
void runParserTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runLexerTests();
    runParserTests();
    std::cout << "ForgeCC unit tests passed\n";
}
