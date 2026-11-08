#include "forgecc/memory/memory_plan.hpp"

#include <cassert>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::ir::ValueId;
using forgecc::memory::BufferId;
using forgecc::memory::MemoryPlan;
using forgecc::memory::MemoryPlanner;
using forgecc::memory::LivenessAnalyzer;

forgecc::semantic::TensorType f32(std::vector<std::size_t> dimensions) {
    return forgecc::semantic::TensorType{forgecc::semantic::ElementType::F32,
                                         std::move(dimensions)};
}

Graph twoIndependentRelus(std::vector<std::size_t> firstShape,
                          std::vector<std::size_t> secondShape) {
    Graph graph;
    const ValueId firstInput = graph.addInput("A", f32(firstShape));
    const ValueId secondInput = graph.addInput("B", f32(secondShape));
    const auto first =
        graph.addNode(OperationKind::Relu, {firstInput}, f32(std::move(firstShape)));
    const auto second = graph.addNode(OperationKind::Relu, {secondInput},
                                      f32(std::move(secondShape)));
    assert(first == 0);
    graph.addOutput(graph.node(second).output);
    return graph;
}

void assertValidPlan(const Graph& graph, const MemoryPlan& plan) {
    forgecc::common::DiagnosticEngine diagnostics;
    assert(plan.validate(graph, diagnostics));
    assert(!diagnostics.hasErrors());
}

void testLifetimeAnalysisTracksProductionAndFinalConsumption() {
    Graph graph;
    const ValueId input = graph.addInput("A", f32({2}));
    const auto first = graph.addNode(OperationKind::Relu, {input}, f32({2}));
    const auto second = graph.addNode(OperationKind::Relu, {graph.node(first).output},
                                     f32({2}));
    graph.addOutput(graph.node(second).output);

    LivenessAnalyzer analyzer;
    auto analysis = analyzer.analyze(graph);
    assert(analysis.has_value());
    assert(analysis->executionOrder == std::vector<forgecc::ir::NodeId>({0, 1}));
    assert(analysis->values.at(input).isGraphInput);
    assert(!analysis->values.at(input).productionIndex.has_value());
    assert(analysis->values.at(input).finalConsumptionIndex == 0);
    assert(analysis->values.at(graph.node(first).output).productionIndex == 0);
    assert(analysis->values.at(graph.node(first).output).finalConsumptionIndex == 1);
    assert(analysis->values.at(graph.node(second).output).isGraphOutput);
    assert(analysis->values.at(graph.node(second).output).productionIndex == 1);
}

void testOverlappingLifetimesUseDifferentBuffers() {
    Graph graph;
    const ValueId firstInput = graph.addInput("A", f32({2}));
    const ValueId secondInput = graph.addInput("B", f32({2}));
    const auto first = graph.addNode(OperationKind::Relu, {firstInput}, f32({2}));
    const auto second = graph.addNode(OperationKind::Relu, {secondInput}, f32({2}));
    const auto firstOutput = graph.addNode(OperationKind::Relu, {graph.node(first).output},
                                           f32({2}));
    const auto secondOutput =
        graph.addNode(OperationKind::Relu, {graph.node(second).output}, f32({2}));
    graph.addOutput(graph.node(firstOutput).output);
    graph.addOutput(graph.node(secondOutput).output);

    MemoryPlanner planner;
    auto plan = planner.plan(graph);
    assert(plan.has_value());
    assert(plan->valueBuffers.at(graph.node(first).output) !=
           plan->valueBuffers.at(graph.node(second).output));
    assertValidPlan(graph, *plan);

    MemoryPlan invalid = *plan;
    invalid.valueBuffers.at(graph.node(second).output) =
        invalid.valueBuffers.at(graph.node(first).output);
    forgecc::common::DiagnosticEngine diagnostics;
    assert(!invalid.validate(graph, diagnostics));
    assert(diagnostics.hasErrors());
}

void testNonOverlappingLifetimesReuseBuffers() {
    Graph graph = twoIndependentRelus({2}, {2});
    MemoryPlanner planner;
    auto plan = planner.plan(graph);

    assert(plan.has_value());
    assert(plan->valueBuffers.at(0) == std::nullopt);
    assert(plan->valueBuffers.at(1) == std::nullopt);
    assert(plan->valueBuffers.at(2) == BufferId{0});
    assert(plan->valueBuffers.at(3) == BufferId{0});
    assert(plan->buffers.at(0).capacity == 2);
    assertValidPlan(graph, *plan);
}

void testBufferCapacityControlsReuse() {
    Graph largerThenSmaller = twoIndependentRelus({4}, {2});
    MemoryPlanner planner;
    auto reusable = planner.plan(largerThenSmaller);
    assert(reusable.has_value());
    assert(reusable->valueBuffers.at(2) == BufferId{0});
    assert(reusable->valueBuffers.at(3) == BufferId{0});
    assert(reusable->buffers.at(0).capacity == 4);

    Graph smallerThenLarger = twoIndependentRelus({2}, {4});
    auto separate = planner.plan(smallerThenLarger);
    assert(separate.has_value());
    assert(separate->valueBuffers.at(2) == BufferId{0});
    assert(separate->valueBuffers.at(3) == BufferId{1});
    assert(separate->buffers.at(1).capacity == 4);
    assertValidPlan(largerThenSmaller, *reusable);
    assertValidPlan(smallerThenLarger, *separate);
}

void testInputsAndOutputsHaveCorrectStorageRules() {
    Graph graph;
    const ValueId firstInput = graph.addInput("A", f32({2}));
    const ValueId secondInput = graph.addInput("B", f32({2}));
    const auto first = graph.addNode(OperationKind::Relu, {firstInput}, f32({2}));
    const auto second = graph.addNode(OperationKind::Relu, {secondInput}, f32({2}));
    graph.addOutput(graph.node(first).output);
    graph.addOutput(graph.node(second).output);

    MemoryPlanner planner;
    auto plan = planner.plan(graph);
    assert(plan.has_value());
    assert(plan->valueBuffers.at(firstInput) == std::nullopt);
    assert(plan->valueBuffers.at(secondInput) == std::nullopt);
    assert(plan->valueBuffers.at(graph.node(first).output).has_value());
    assert(plan->valueBuffers.at(graph.node(second).output).has_value());
    assert(plan->valueBuffers.at(graph.node(first).output) !=
           plan->valueBuffers.at(graph.node(second).output));
    assert(plan->lifetimes.at(graph.node(first).output).isGraphOutput);
    assert(plan->lifetimes.at(graph.node(second).output).isGraphOutput);
    assertValidPlan(graph, *plan);
}

void testMultipleConsumersExtendLifetime() {
    Graph graph;
    const ValueId input = graph.addInput("A", f32({2}));
    const auto source = graph.addNode(OperationKind::Relu, {input}, f32({2}));
    const auto firstConsumer = graph.addNode(OperationKind::Relu,
                                             {graph.node(source).output}, f32({2}));
    const auto secondConsumer = graph.addNode(OperationKind::Relu,
                                              {graph.node(source).output}, f32({2}));
    graph.addOutput(graph.node(firstConsumer).output);
    graph.addOutput(graph.node(secondConsumer).output);

    MemoryPlanner planner;
    auto plan = planner.plan(graph);
    assert(plan.has_value());
    const auto& lifetime = plan->lifetimes.at(graph.node(source).output);
    assert(lifetime.finalConsumptionIndex == 2);
    assert(plan->valueBuffers.at(graph.node(source).output) !=
           plan->valueBuffers.at(graph.node(firstConsumer).output));
    assertValidPlan(graph, *plan);
}

void testAllocationResultsAreDeterministic() {
    Graph graph = twoIndependentRelus({4}, {2});
    MemoryPlanner firstPlanner;
    MemoryPlanner secondPlanner;
    auto first = firstPlanner.plan(graph);
    auto second = secondPlanner.plan(graph);

    assert(first.has_value());
    assert(second.has_value());
    assert(first->lifetimes == second->lifetimes);
    assert(first->elementCounts == second->elementCounts);
    assert(first->valueBuffers == second->valueBuffers);
    assert(first->buffers.at(0).capacity == second->buffers.at(0).capacity);
}

void testInvalidShapesAndOverflowFailSafely() {
    Graph emptyShape;
    const ValueId emptyInput = emptyShape.addInput("A", f32({}));
    assert(emptyInput == 0);
    MemoryPlanner planner;
    assert(!planner.plan(emptyShape).has_value());
    assert(planner.diagnostics().hasErrors());

    Graph overflow;
    const ValueId overflowInput =
        overflow.addInput("A", f32({std::numeric_limits<std::size_t>::max(), 2}));
    assert(overflowInput == 0);
    assert(!planner.plan(overflow).has_value());
    assert(planner.diagnostics().hasErrors());

    Graph invalidReference;
    const auto node = invalidReference.addNode(OperationKind::Relu, {42}, f32({2}));
    invalidReference.addOutput(invalidReference.node(node).output);
    assert(!planner.plan(invalidReference).has_value());
    assert(planner.diagnostics().hasErrors());
}

} // namespace

void runMemoryPlanTests() {
    testLifetimeAnalysisTracksProductionAndFinalConsumption();
    testOverlappingLifetimesUseDifferentBuffers();
    testNonOverlappingLifetimesReuseBuffers();
    testBufferCapacityControlsReuse();
    testInputsAndOutputsHaveCorrectStorageRules();
    testMultipleConsumersExtendLifetime();
    testAllocationResultsAreDeterministic();
    testInvalidShapesAndOverflowFailSafely();
}
