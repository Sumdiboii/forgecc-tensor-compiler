#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/frontend/ast.hpp"
#include "forgecc/semantic/symbol_table.hpp"

namespace forgecc::frontend {

class SemanticAnalyzer {
public:
    [[nodiscard]] bool analyze(const Program& program);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;
    [[nodiscard]] const semantic::SymbolTable& symbols() const;

private:
    void analyzeStatement(const Statement& statement);
    void analyzeDeclaration(const TensorDeclarationStatement& declaration);
    void analyzeAssignment(const AssignmentStatement& assignment);
    void analyzeExpression(const Expression& expression);
    void analyzeCall(const CallExpression& call);
    void reportError(common::SourceLocation location, std::string message);

    semantic::SymbolTable symbols_;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::frontend
