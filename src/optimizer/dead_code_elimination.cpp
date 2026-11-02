#include "forgecc/optimizer/dead_code_elimination.hpp"

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace forgecc::optimizer {

std::string_view DeadCodeElimination::name() const {
    return "dead-code-elimination";
}

PassResult DeadCodeElimination::run(ir::Graph& graph,
                                    common::DiagnosticEngine& diagnostics) {
    if (!graph.validate(diagnostics)) {
        return PassResult{};
    }

    std::unordered_set<ir::NodeId> live;
    const std::function<void(ir::ValueId)> markValue = [&](ir::ValueId valueId) {
        const ir::Value& value = graph.value(valueId);
        if (!value.producer.has_value() || !graph.isNodeLive(*value.producer) ||
            !live.insert(*value.producer).second) {
            return;
        }

        const ir::Node& node = graph.node(*value.producer);
        for (ir::ValueId input : node.inputs) {
            markValue(input);
        }
    };

    for (ir::ValueId output : graph.outputs()) {
        markValue(output);
    }

    std::vector<ir::NodeId> liveNodes(live.begin(), live.end());
    std::sort(liveNodes.begin(), liveNodes.end());
    const bool changed = graph.keepOnlyNodes(liveNodes);
    if (!graph.validate(diagnostics)) {
        return PassResult{};
    }

    return PassResult{changed};
}

} // namespace forgecc::optimizer
