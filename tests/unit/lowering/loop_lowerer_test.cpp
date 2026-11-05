#include "forgecc/ir/graph.hpp"
#include "forgecc/lowering/loop_lowerer.hpp"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace {

using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::ir::ValueId;
using forgecc::lowering::BinaryExpression;
using forgecc::lowering::BinaryOperator;
using forgecc::lowering::Expression;
using forgecc::lowering::Index;
using forgecc::lowering::Loop;
using forgecc::lowering::LoopLowerer;
using forgecc::lowering::LoopProgram;
using forgecc::lowering::UnaryExpression;

forgecc::semantic::TensorType f32(std::vector<std::size_t> dimensions) {
    return forgecc::semantic::TensorType{forgecc::semantic::ElementType::F32,
                                         std::move(dimensions)};
}

const Loop& nestedLoop(const forgecc::lowering::Block& block, std::size_t index) {
    const auto* loop = std::get_if<forgecc::lowering::LoopPtr>(
        &block.instructions[index].storage);
    assert(loop != nullptr);
    assert(*loop != nullptr);
    return **loop;
}

Graph makeReluGraph() {
    Graph graph;
    const ValueId input = graph.addInput("A", f32({2, 3}));
    const auto node = graph.addNode(OperationKind::Relu, {input}, f32({2, 3}));
    graph.setValueName(graph.node(node).output, "C");
    graph.addOutput(graph.node(node).output);
    return graph;
}

Graph makeMatmulGraph() {
    Graph graph;
    const ValueId left = graph.addInput("A", f32({2, 3}));
    const ValueId right = graph.addInput("B", f32({3, 4}));
    const auto node = graph.addNode(OperationKind::Matmul, {left, right}, f32({2, 4}));
    graph.setValueName(graph.node(node).output, "C");
    graph.addOutput(graph.node(node).output);
    return graph;
}

void testReluLoweringProducesElementwiseLoops() {
    LoopLowerer lowerer;
    auto program = lowerer.lower(makeReluGraph());

    assert(program.has_value());
    assert(program->tensorNames.at(0) == "A");
    assert(program->tensorNames.at(1) == "C");
    const Loop& outer = nestedLoop(program->body, 0);
    assert(outer.variable == "i0");
    assert(outer.bound == 2);
    const Loop& inner = nestedLoop(outer.body, 0);
    assert(inner.variable == "i1");
    assert(inner.bound == 3);

    const auto* store = std::get_if<forgecc::lowering::StoreInstruction>(
        &inner.body.instructions.front().storage);
    assert(store != nullptr);
    assert(store->target.tensor == "C");
    assert(store->target.indices.size() == 2);
    assert(store->target.indices[0].kind == Index::Kind::IterationVariable);
    assert(store->target.indices[0].variable == "i0");
    assert(store->target.indices[1].kind == Index::Kind::IterationVariable);
    assert(store->target.indices[1].variable == "i1");
    const auto* unary = std::get_if<UnaryExpression>(&store->value->storage);
    assert(unary != nullptr);
    assert(unary->operation == forgecc::lowering::UnaryOperator::Relu);
    assert(unary->operand != nullptr);
    const auto* input = std::get_if<forgecc::lowering::TensorElement>(
        &unary->operand->storage);
    assert(input != nullptr);
    assert(input->tensor == "A");
    assert(input->indices.size() == 2);
    assert(input->indices[0].kind == Index::Kind::IterationVariable);
    assert(input->indices[0].variable == "i0");
    assert(input->indices[1].kind == Index::Kind::IterationVariable);
    assert(input->indices[1].variable == "i1");
}

void testMatmulLoweringInitializesAndAccumulates() {
    LoopLowerer lowerer;
    auto program = lowerer.lower(makeMatmulGraph());

    assert(program.has_value());
    assert(program->tensorNames.at(0) == "A");
    assert(program->tensorNames.at(1) == "B");
    assert(program->tensorNames.at(2) == "C");

    const Loop& outer = nestedLoop(program->body, 0);
    const Loop& middle = nestedLoop(outer.body, 0);
    assert(outer.variable == "i");
    assert(outer.bound == 2);
    assert(middle.variable == "j");
    assert(middle.bound == 4);
    assert(middle.body.instructions.size() == 2);

    const auto* initialization = std::get_if<forgecc::lowering::StoreInstruction>(
        &middle.body.instructions[0].storage);
    assert(initialization != nullptr);
    assert(initialization->target.tensor == "C");
    assert(initialization->target.indices.size() == 2);
    assert(initialization->target.indices[0].variable == "i");
    assert(initialization->target.indices[1].variable == "j");
    assert(std::get<double>(initialization->value->storage) == 0.0);

    const Loop& reduction = nestedLoop(middle.body, 1);
    assert(reduction.variable == "k");
    assert(reduction.bound == 3);
    const auto* accumulation = std::get_if<forgecc::lowering::StoreInstruction>(
        &reduction.body.instructions.front().storage);
    assert(accumulation != nullptr);
    const auto* addition = std::get_if<BinaryExpression>(&accumulation->value->storage);
    assert(addition != nullptr);
    assert(addition->operation == BinaryOperator::Add);
    const auto* existingOutput = std::get_if<forgecc::lowering::TensorElement>(
        &addition->left->storage);
    assert(existingOutput != nullptr);
    assert(existingOutput->tensor == "C");
    assert(existingOutput->indices.size() == 2);
    assert(existingOutput->indices[0].variable == "i");
    assert(existingOutput->indices[1].variable == "j");
    const auto* product = std::get_if<BinaryExpression>(&addition->right->storage);
    assert(product != nullptr);
    assert(product->operation == BinaryOperator::Multiply);
    const auto& left = std::get<forgecc::lowering::TensorElement>(product->left->storage);
    const auto& right = std::get<forgecc::lowering::TensorElement>(product->right->storage);
    assert(left.tensor == "A");
    assert(left.indices.size() == 2);
    assert(left.indices[0].variable == "i");
    assert(left.indices[1].variable == "k");
    assert(right.tensor == "B");
    assert(right.indices.size() == 2);
    assert(right.indices[0].variable == "k");
    assert(right.indices[1].variable == "j");
}

void testMultipleOperationsPreserveDependencyOrderAndMapping() {
    Graph graph;
    const ValueId left = graph.addInput("A", f32({2, 3}));
    const ValueId right = graph.addInput("B", f32({3, 4}));
    const auto matmul = graph.addNode(OperationKind::Matmul, {left, right}, f32({2, 4}));
    graph.setValueName(graph.node(matmul).output, "M");
    const auto relu = graph.addNode(OperationKind::Relu, {graph.node(matmul).output},
                                    f32({2, 4}));
    graph.setValueName(graph.node(relu).output, "C");
    graph.addOutput(graph.node(relu).output);

    LoopLowerer lowerer;
    auto program = lowerer.lower(graph);
    assert(program.has_value());
    assert(program->body.instructions.size() == 2);
    assert(program->tensorNames.at(2) == "M");
    assert(program->tensorNames.at(3) == "C");
    assert(nestedLoop(program->body, 0).variable == "i");
    assert(nestedLoop(program->body, 1).variable == "i0");
}

void testLoweringRejectsInvalidShapesAndGraphReferences() {
    Graph invalidRank;
    const ValueId vector = invalidRank.addInput("A", f32({2}));
    const ValueId matrix = invalidRank.addInput("B", f32({2, 3}));
    const auto rankNode = invalidRank.addNode(OperationKind::Matmul, {vector, matrix},
                                              f32({2, 3}));
    invalidRank.addOutput(invalidRank.node(rankNode).output);
    LoopLowerer lowerer;
    assert(!lowerer.lower(invalidRank).has_value());
    assert(lowerer.diagnostics().hasErrors());

    Graph incompatible;
    const ValueId first = incompatible.addInput("A", f32({2, 3}));
    const ValueId second = incompatible.addInput("B", f32({4, 2}));
    const auto incompatibleNode =
        incompatible.addNode(OperationKind::Matmul, {first, second}, f32({2, 2}));
    incompatible.addOutput(incompatible.node(incompatibleNode).output);
    assert(!lowerer.lower(incompatible).has_value());
    assert(lowerer.diagnostics().hasErrors());

    Graph invalidReference;
    const ValueId input = invalidReference.addInput("A", f32({2, 2}));
    const auto invalidNode = invalidReference.addNode(OperationKind::Relu, {input + 10},
                                                      f32({2, 2}));
    invalidReference.addOutput(invalidReference.node(invalidNode).output);
    assert(!lowerer.lower(invalidReference).has_value());
}

void testLoweringRejectsUnsupportedOperationsAndNameCollisions() {
    Graph unsupported;
    const auto node = unsupported.addNode(static_cast<OperationKind>(99), {}, f32({2}));
    unsupported.addOutput(unsupported.node(node).output);
    LoopLowerer lowerer;
    assert(!lowerer.lower(unsupported).has_value());
    assert(lowerer.diagnostics().hasErrors());

    Graph duplicateNames;
    const ValueId first = duplicateNames.addInput("A", f32({2}));
    const ValueId second = duplicateNames.addInput("A", f32({2}));
    assert(first == 0);
    assert(second == 1);
    assert(!lowerer.lower(duplicateNames).has_value());
}

} // namespace

void runLoopLowererTests() {
    testReluLoweringProducesElementwiseLoops();
    testMatmulLoweringInitializesAndAccumulates();
    testMultipleOperationsPreserveDependencyOrderAndMapping();
    testLoweringRejectsInvalidShapesAndGraphReferences();
    testLoweringRejectsUnsupportedOperationsAndNameCollisions();
}
