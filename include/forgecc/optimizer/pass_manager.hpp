#pragma once

#include "forgecc/optimizer/pass.hpp"

#include <memory>
#include <vector>

namespace forgecc::optimizer {

struct PassManagerResult {
    bool success = false;
    bool changed = false;
};

class PassManager {
public:
    void addPass(std::unique_ptr<OptimizationPass> pass);

    [[nodiscard]] PassManagerResult run(ir::Graph& graph,
                                        common::DiagnosticEngine& diagnostics) const;

private:
    std::vector<std::unique_ptr<OptimizationPass>> passes_;
};

} // namespace forgecc::optimizer
