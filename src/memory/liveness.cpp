#include "forgecc/memory/liveness.hpp"

#include <utility>

namespace forgecc::memory {

namespace {

void report(common::DiagnosticEngine& diagnostics, std::string message) {
    diagnostics.reportError({1, 1}, std::move(message));
}

} // namespace

std::optional<LifetimeAnalysis> LivenessAnalyzer::analyze(const ir::Graph& graph) {
    diagnostics_ = common::DiagnosticEngine{};

    auto order = graph.topologicalOrder(diagnostics_);
    if (!order.has_value()) {
        return std::nullopt;
    }

    LifetimeAnalysis analysis;
    analysis.executionOrder = *order;
    for (ir::ValueId value = 0; value < graph.valueCount(); ++value) {
        const ir::Value& graphValue = graph.value(value);
        analysis.values.emplace(
            value, ValueLifetime{value, graphValue.kind == ir::ValueKind::GraphInput, false,
                                 std::nullopt, std::nullopt});
    }

    for (std::size_t index = 0; index < analysis.executionOrder.size(); ++index) {
        const ir::Node& node = graph.node(analysis.executionOrder[index]);
        const auto output = analysis.values.find(node.output);
        if (output == analysis.values.end()) {
            report(diagnostics_, "node output is missing from lifetime analysis");
            return std::nullopt;
        }
        output->second.productionIndex = index;

        for (ir::ValueId input : node.inputs) {
            const auto lifetime = analysis.values.find(input);
            if (lifetime == analysis.values.end()) {
                report(diagnostics_, "node input is missing from lifetime analysis");
                return std::nullopt;
            }
            if (!lifetime->second.finalConsumptionIndex.has_value() ||
                *lifetime->second.finalConsumptionIndex < index) {
                lifetime->second.finalConsumptionIndex = index;
            }
        }
    }

    for (ir::ValueId output : graph.outputs()) {
        const auto lifetime = analysis.values.find(output);
        if (lifetime == analysis.values.end()) {
            report(diagnostics_, "graph output is missing from lifetime analysis");
            return std::nullopt;
        }
        lifetime->second.isGraphOutput = true;
    }

    return analysis;
}

const common::DiagnosticEngine& LivenessAnalyzer::diagnostics() const {
    return diagnostics_;
}

} // namespace forgecc::memory
