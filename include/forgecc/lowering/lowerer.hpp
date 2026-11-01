#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/frontend/ast.hpp"
#include "forgecc/frontend/semantic_analyzer.hpp"
#include "forgecc/ir/graph.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace forgecc::lowering {

class GraphLowerer {
public:
    [[nodiscard]] std::optional<ir::Graph> lower(
        const frontend::Program& program, frontend::SemanticAnalyzer& analyzer);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    [[nodiscard]] std::optional<ir::ValueId> lowerExpression(
        const frontend::Expression& expression);
    [[nodiscard]] std::optional<ir::ValueId> lowerCall(
        const frontend::CallExpression& call);
    void reportError(common::SourceLocation location, std::string message);

    frontend::SemanticAnalyzer* analyzer_ = nullptr;
    ir::Graph graph_;
    std::unordered_map<std::string, ir::ValueId> values_;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::lowering
