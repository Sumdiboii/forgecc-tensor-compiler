#include "forgecc/memory/memory_plan.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace forgecc::memory {

namespace {

void report(common::DiagnosticEngine& diagnostics, std::string message) {
    diagnostics.reportError({1, 1}, std::move(message));
}

bool checkedElementCount(const semantic::TensorType& type, std::size_t& count,
                         common::DiagnosticEngine& diagnostics) {
    if (type.elementType != semantic::ElementType::F32) {
        report(diagnostics, "memory planning supports only f32 tensors");
        return false;
    }
    if (type.dimensions.empty()) {
        report(diagnostics, "tensor shape must contain at least one dimension");
        return false;
    }

    count = 1;
    for (std::size_t dimension : type.dimensions) {
        if (count != 0 && dimension > std::numeric_limits<std::size_t>::max() / count) {
            report(diagnostics, "tensor element count overflows size_t");
            return false;
        }
        count *= dimension;
    }
    return true;
}

std::size_t lifetimeEnd(const ValueLifetime& lifetime) {
    if (lifetime.isGraphOutput) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lifetime.finalConsumptionIndex.value_or(
        lifetime.productionIndex.value_or(std::size_t{0}));
}

bool intervalsOverlap(const ValueLifetime& left, const ValueLifetime& right) {
    const std::size_t leftStart = left.productionIndex.value_or(0);
    const std::size_t rightStart = right.productionIndex.value_or(0);
    return leftStart <= lifetimeEnd(right) && rightStart <= lifetimeEnd(left);
}

} // namespace

bool MemoryPlan::validate(const ir::Graph& graph,
                          common::DiagnosticEngine& diagnostics) const {
    bool valid = true;
    const auto reportInvalid = [&diagnostics, &valid](std::string message) {
        diagnostics.reportError({1, 1}, std::move(message));
        valid = false;
    };

    if (!graph.validate(diagnostics)) {
        return false;
    }

    for (ir::ValueId value = 0; value < graph.valueCount(); ++value) {
        const auto lifetime = lifetimes.find(value);
        const auto count = elementCounts.find(value);
        const auto allocation = valueBuffers.find(value);
        if (lifetime == lifetimes.end() || count == elementCounts.end() ||
            allocation == valueBuffers.end()) {
            reportInvalid("memory plan is missing metadata for a graph value");
            continue;
        }

        const ir::Value& graphValue = graph.value(value);
        std::size_t expectedCount = 0;
        if (!checkedElementCount(graphValue.type, expectedCount, diagnostics)) {
            valid = false;
        } else if (count->second != expectedCount) {
            reportInvalid("memory plan element count does not match tensor shape");
        }

        const bool isGraphOutput = std::find(graph.outputs().begin(), graph.outputs().end(),
                                             value) != graph.outputs().end();
        if (lifetime->second.isGraphOutput != isGraphOutput) {
            reportInvalid("memory plan graph-output classification is inconsistent");
        }

        if (graphValue.kind == ir::ValueKind::GraphInput) {
            if (!lifetime->second.isGraphInput) {
                reportInvalid("graph input is missing its lifetime classification");
            }
            if (lifetime->second.productionIndex.has_value()) {
                reportInvalid("graph input cannot have a production point");
            }
            if (allocation->second.has_value()) {
                reportInvalid("graph input has an intermediate buffer allocation");
            }
            continue;
        }
        if (lifetime->second.isGraphInput) {
            reportInvalid("operation output is incorrectly classified as a graph input");
        }
        if (!lifetime->second.productionIndex.has_value()) {
            if (allocation->second.has_value()) {
                reportInvalid("unexecuted graph value has a buffer allocation");
            }
            continue;
        }
        if (*lifetime->second.productionIndex >= graph.nodeCount()) {
            reportInvalid("graph value production point is outside execution order");
        }
        if (lifetime->second.finalConsumptionIndex.has_value()) {
            if (*lifetime->second.finalConsumptionIndex >= graph.nodeCount()) {
                reportInvalid("graph value consumption point is outside execution order");
            }
            if (*lifetime->second.finalConsumptionIndex <
                *lifetime->second.productionIndex) {
                reportInvalid("graph value is consumed before it is produced");
            }
        }
        if (!allocation->second.has_value()) {
            reportInvalid("intermediate graph value has no buffer allocation");
            continue;
}

        const auto buffer = buffers.find(*allocation->second);
        if (buffer == buffers.end()) {
            reportInvalid("graph value refers to an invalid buffer allocation");
        } else if (buffer->second.id != buffer->first) {
            reportInvalid("buffer allocation ID does not match its map entry");
        } else if (buffer->second.capacity < count->second) {
            reportInvalid("buffer capacity is smaller than tensor element count");
        }
    }

    for (const auto& [id, buffer] : buffers) {
        if (id != buffer.id) {
            reportInvalid("buffer allocation ID does not match its map entry");
        }
    }

    for (ir::ValueId output : graph.outputs()) {
        const auto lifetime = lifetimes.find(output);
        if (lifetime == lifetimes.end() || !lifetime->second.isGraphOutput) {
            reportInvalid("graph output is missing its lifetime classification");
        }
    }

    for (const auto& [leftValue, leftAllocation] : valueBuffers) {
        if (!leftAllocation.has_value()) {
            continue;
        }
        const auto leftLifetime = lifetimes.find(leftValue);
        if (leftLifetime == lifetimes.end() ||
            !leftLifetime->second.productionIndex.has_value()) {
            continue;
        }
        for (const auto& [rightValue, rightAllocation] : valueBuffers) {
            if (leftValue >= rightValue || !rightAllocation.has_value() ||
                *leftAllocation != *rightAllocation) {
                continue;
            }
            const auto rightLifetime = lifetimes.find(rightValue);
            if (rightLifetime != lifetimes.end() &&
                rightLifetime->second.productionIndex.has_value() &&
                intervalsOverlap(leftLifetime->second, rightLifetime->second)) {
                reportInvalid("overlapping tensor lifetimes share a buffer");
            }
        }
    }

    return valid;
}

std::optional<MemoryPlan> MemoryPlanner::plan(const ir::Graph& graph) {
    diagnostics_ = common::DiagnosticEngine{};

    LivenessAnalyzer analyzer;
    auto analysis = analyzer.analyze(graph);
    if (!analysis.has_value()) {
        for (const common::Diagnostic& diagnostic : analyzer.diagnostics().diagnostics()) {
            diagnostics_.report(diagnostic.severity, diagnostic.location, diagnostic.message);
        }
        return std::nullopt;
    }

    MemoryPlan plan;
    plan.lifetimes = analysis->values;
    for (ir::ValueId value = 0; value < graph.valueCount(); ++value) {
        std::size_t count = 0;
        if (!checkedElementCount(graph.value(value).type, count, diagnostics_)) {
            return std::nullopt;
        }
        plan.elementCounts.emplace(value, count);
        plan.valueBuffers.emplace(value, std::nullopt);
    }

    struct BufferState {
        BufferAllocation allocation;
        std::optional<ir::ValueId> lastValue;
    };
    std::vector<BufferState> bufferStates;

    for (std::size_t index = 0; index < analysis->executionOrder.size(); ++index) {
        const ir::Node& node = graph.node(analysis->executionOrder[index]);
        const auto lifetime = analysis->values.find(node.output);
        if (lifetime == analysis->values.end() || !lifetime->second.productionIndex.has_value()) {
            report(diagnostics_, "operation output is missing a production point");
            return std::nullopt;
        }

        const std::size_t required = plan.elementCounts.at(node.output);
        BufferState* selected = nullptr;
        for (BufferState& state : bufferStates) {
            if (state.allocation.capacity < required || !state.lastValue.has_value()) {
                if (!state.lastValue.has_value() && state.allocation.capacity >= required) {
                    selected = &state;
                    break;
                }
                continue;
            }

            const ValueLifetime& previous = analysis->values.at(*state.lastValue);
            if (!previous.isGraphOutput && lifetimeEnd(previous) < index) {
                selected = &state;
                break;
            }
        }

        if (selected == nullptr) {
            const BufferId id = bufferStates.size();
            bufferStates.push_back(BufferState{BufferAllocation{id, required}, std::nullopt});
            selected = &bufferStates.back();
        }
        selected->allocation.capacity = std::max(selected->allocation.capacity, required);
        selected->lastValue = node.output;
        plan.valueBuffers[node.output] = selected->allocation.id;
    }

    for (const BufferState& state : bufferStates) {
        plan.buffers.emplace(state.allocation.id, state.allocation);
    }

    if (!plan.validate(graph, diagnostics_)) {
        return std::nullopt;
    }
    return plan;
}

const common::DiagnosticEngine& MemoryPlanner::diagnostics() const {
    return diagnostics_;
}

} // namespace forgecc::memory
