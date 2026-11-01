#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/frontend/ast.hpp"
#include "forgecc/semantic/symbol_table.hpp"

#include <optional>

namespace forgecc::frontend {

class SemanticAnalyzer {
public:
    [[nodiscard]] bool analyze(const Program& program);
    [[nodiscard]] std::optional<semantic::TensorType> inferShape(const Expression& expression);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;
    [[nodiscard]] const semantic::SymbolTable& symbols() const;

private:
    void analyzeStatement(const Statement& statement);
    void analyzeDeclaration(const TensorDeclarationStatement& declaration);
    void analyzeAssignment(const AssignmentStatement& assignment);
    [[nodiscard]] std::optional<semantic::TensorType> inferExpression(
        const Expression& expression);
    [[nodiscard]] std::optional<semantic::TensorType> inferCall(const CallExpression& call);
    [[nodiscard]] std::optional<semantic::TensorType> inferBinary(
        const BinaryExpression& binary);
    void reportError(common::SourceLocation location, std::string message);

    semantic::SymbolTable symbols_;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::frontend
