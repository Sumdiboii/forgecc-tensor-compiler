#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/graph.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace forgecc::memory {

struct ValueLifetime {
    ir::ValueId value;
    bool isGraphInput = false;
    bool isGraphOutput = false;
    // Indices refer to positions in LifetimeAnalysis::executionOrder.
    // Graph inputs have no production point; graph outputs remain live after execution.
    std::optional<std::size_t> productionIndex;
    std::optional<std::size_t> finalConsumptionIndex;

    bool operator==(const ValueLifetime&) const = default;
};

struct LifetimeAnalysis {
    std::vector<ir::NodeId> executionOrder;
    std::map<ir::ValueId, ValueLifetime> values;
};

class LivenessAnalyzer {
public:
    [[nodiscard]] std::optional<LifetimeAnalysis> analyze(const ir::Graph& graph);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::memory
