#pragma once

#include "forgecc/optimizer/pass.hpp"

namespace forgecc::optimizer {

class CommonSubexpressionElimination final : public OptimizationPass {
public:
    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] PassResult run(ir::Graph& graph,
                                 common::DiagnosticEngine& diagnostics) override;
};

} // namespace forgecc::optimizer
