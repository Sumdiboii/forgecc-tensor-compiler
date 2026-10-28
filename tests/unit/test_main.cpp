#include <iostream>

void runDiagnosticTests();
void runSourceLocationTests();
void runLexerTests();

int main() {
    runDiagnosticTests();
    runSourceLocationTests();
    runLexerTests();
    std::cout << "ForgeCC unit tests passed\n";
}
