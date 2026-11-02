#include "forgecc/optimizer/pass_manager.hpp"

namespace forgecc::optimizer {

void PassManager::addPass(std::unique_ptr<OptimizationPass> pass) {
    passes_.push_back(std::move(pass));
}

PassManagerResult PassManager::run(ir::Graph& graph,
                                   common::DiagnosticEngine& diagnostics) const {
    PassManagerResult result{true, false};
    if (!graph.validate(diagnostics)) {
        result.success = false;
        return result;
    }

    for (const std::unique_ptr<OptimizationPass>& pass : passes_) {
        const PassResult passResult = pass->run(graph, diagnostics);
        if (diagnostics.hasErrors() || !graph.validate(diagnostics)) {
            result.success = false;
            return result;
        }
        result.changed = result.changed || passResult.changed;
    }

    return result;
}

} // namespace forgecc::optimizer
