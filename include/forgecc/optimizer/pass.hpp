#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/graph.hpp"

#include <string_view>

namespace forgecc::optimizer {

struct PassResult {
    bool changed = false;
};

class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual PassResult run(ir::Graph& graph,
                                         common::DiagnosticEngine& diagnostics) = 0;
};

} // namespace forgecc::optimizer
