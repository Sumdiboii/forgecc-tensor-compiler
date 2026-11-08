#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/graph.hpp"
#include "forgecc/memory/liveness.hpp"

#include <cstddef>
#include <map>
#include <optional>

namespace forgecc::memory {

using BufferId = std::size_t;

struct BufferAllocation {
    BufferId id;
    std::size_t capacity;

    bool operator==(const BufferAllocation&) const = default;
};

struct MemoryPlan {
    // Logical graph values remain separate from these physical buffer IDs.
    std::map<ir::ValueId, ValueLifetime> lifetimes;
    std::map<ir::ValueId, std::size_t> elementCounts;
    std::map<ir::ValueId, std::optional<BufferId>> valueBuffers;
    std::map<BufferId, BufferAllocation> buffers;

    [[nodiscard]] bool validate(const ir::Graph& graph,
                                common::DiagnosticEngine& diagnostics) const;
};

class MemoryPlanner {
public:
    [[nodiscard]] std::optional<MemoryPlan> plan(const ir::Graph& graph);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::memory
