#include "forgecc/lowering/loop_ir.hpp"

#include <cassert>
#include <memory>
#include <vector>

namespace {

using forgecc::common::DiagnosticEngine;
using forgecc::lowering::BinaryOperator;
using forgecc::lowering::Block;
using forgecc::lowering::Expression;
using forgecc::lowering::Index;
using forgecc::lowering::Instruction;
using forgecc::lowering::Loop;
using forgecc::lowering::TensorElement;
using forgecc::lowering::UnaryOperator;

TensorElement element(const char* tensor, std::vector<Index> indices) {
    return TensorElement{tensor, std::move(indices)};
}

bool validatesElement(TensorElement tensor) {
    Block block;
    block.appendStore(std::move(tensor),
                      std::make_unique<Expression>(Expression::constant(1.0)));

    DiagnosticEngine diagnostics;
    return block.validate(diagnostics);
}

std::size_t iterationCount(const Loop& loop) {
    if (loop.start >= loop.bound) {
        return 0;
    }

    return 1 + (loop.bound - 1 - loop.start) / loop.step;
}

void testLoopStoresBoundsAndBody() {
    Block body;
    body.appendStore(element("C", {Index::iteration("i"), Index::constantValue(0)}),
                     std::make_unique<Expression>(Expression::constant(1.0)));
    Loop loop{"i", 0, 8, 2, std::move(body)};

    assert(loop.variable == "i");
    assert(loop.start == 0);
    assert(loop.bound == 8);
    assert(loop.step == 2);
    assert(loop.body.instructions.size() == 1);
    assert(std::holds_alternative<forgecc::lowering::StoreInstruction>(
        loop.body.instructions.front().storage));

    DiagnosticEngine diagnostics;
    assert(loop.validate(diagnostics));
}

void testNestedLoopsAreOrderedInstructions() {
    auto inner = std::make_unique<Loop>(Loop{"j", 0, 4, 1, {}});
    Block outerBody;
    outerBody.appendLoop(std::move(inner));
    Loop outer{"i", 0, 2, 1, std::move(outerBody)};

    assert(outer.body.instructions.size() == 1);
    const auto* nested = std::get_if<forgecc::lowering::LoopPtr>(
        &outer.body.instructions.front().storage);
    assert(nested != nullptr);
    assert(*nested != nullptr);
    assert((*nested)->variable == "j");
}

void testExpressionsRepresentAccessesAndArithmetic() {
    auto left = std::make_unique<Expression>(Expression::tensorElement(
        element("A", {Index::iteration("i"), Index::iteration("k")})));
    auto right = std::make_unique<Expression>(Expression::tensorElement(
        element("B", {Index::iteration("k"), Index::iteration("j")})));
    Expression product = Expression::binary(BinaryOperator::Multiply, std::move(left),
                                            std::move(right));
    Expression sum = Expression::binary(
        BinaryOperator::Add, std::make_unique<Expression>(std::move(product)),
        std::make_unique<Expression>(Expression::constant(2.0)));

    assert(std::holds_alternative<forgecc::lowering::BinaryExpression>(sum.storage));
    const auto& binary = std::get<forgecc::lowering::BinaryExpression>(sum.storage);
    assert(binary.operation == BinaryOperator::Add);
    assert(binary.left != nullptr);
    assert(binary.right != nullptr);
}

void testBlockPreservesInstructionOrder() {
    Block block;
    block.appendStore(element("A", {Index::constantValue(0)}),
                      std::make_unique<Expression>(Expression::constant(1.0)));
    block.appendStore(element("B", {Index::constantValue(0)}),
                      std::make_unique<Expression>(Expression::constant(2.0)));

    assert(block.instructions.size() == 2);
    assert(std::holds_alternative<forgecc::lowering::StoreInstruction>(
        block.instructions[0].storage));
    assert(std::holds_alternative<forgecc::lowering::StoreInstruction>(
        block.instructions[1].storage));
    const auto& first = std::get<forgecc::lowering::StoreInstruction>(block.instructions[0].storage);
    const auto& second = std::get<forgecc::lowering::StoreInstruction>(block.instructions[1].storage);
    assert(first.target.tensor == "A");
    assert(second.target.tensor == "B");
}

void testValidationRejectsInvalidLoopAndStore() {
    Block body;
    body.appendStore(element("", {Index::iteration("")}), nullptr);
    Loop loop{"i", 0, 4, 0, std::move(body)};

    DiagnosticEngine diagnostics;
    assert(!loop.validate(diagnostics));
    assert(diagnostics.hasErrors());
}

void testIndexValidationAcceptsSupportedKinds() {
    assert(validatesElement(element("A", {Index::iteration("i")})));
    assert(validatesElement(element("A", {Index::constantValue(0)})));
}

void testIndexValidationRejectsMalformedKindsAndPayloads() {
    assert(!validatesElement(element("A", {Index::iteration("")})));

    Index invalidKind{static_cast<Index::Kind>(99), {}, 0};
    assert(!validatesElement(element("A", {invalidKind})));

    Index iterationWithConstant{Index::Kind::IterationVariable, "i", 1};
    assert(!validatesElement(element("A", {iterationWithConstant})));

    Index constantWithVariable{Index::Kind::Constant, "i", 0};
    assert(!validatesElement(element("A", {constantWithVariable})));
}

void testLoopUsesExclusiveUpperBound() {
    Loop unitStep{"i", 0, 5, 1, {}};
    Loop strideTwo{"i", 0, 5, 2, {}};
    Loop emptyAtBound{"i", 5, 5, 1, {}};
    Loop emptyPastBound{"i", 6, 5, 1, {}};

    assert(iterationCount(unitStep) == 5);
    assert(iterationCount(strideTwo) == 3);
    assert(iterationCount(emptyAtBound) == 0);
    assert(iterationCount(emptyPastBound) == 0);
}

void testValidationRecursesThroughNestedExpressionsAndLoops() {
    auto invalidExpression = std::make_unique<Expression>(Expression::binary(
        BinaryOperator::Add,
        std::make_unique<Expression>(Expression::tensorElement(
            element("A", {Index::iteration("i")}))),
        nullptr));

    Block block;
    block.appendStore(element("C", {Index::iteration("i")}),
                      std::move(invalidExpression));
    block.appendLoop(nullptr);

    DiagnosticEngine diagnostics;
    assert(!block.validate(diagnostics));
    assert(diagnostics.hasErrors());
}

void testValidationRejectsInvalidBinaryOperator() {
    auto left = std::make_unique<Expression>(Expression::tensorElement(
        element("A", {Index::constantValue(0)})));
    auto right = std::make_unique<Expression>(Expression::tensorElement(
        element("B", {Index::constantValue(0)})));
    Expression invalid = Expression::binary(static_cast<BinaryOperator>(99),
                                             std::move(left), std::move(right));

    Block block;
    block.appendStore(element("C", {Index::constantValue(0)}),
                      std::make_unique<Expression>(std::move(invalid)));

    DiagnosticEngine diagnostics;
    assert(!block.validate(diagnostics));
    assert(diagnostics.hasErrors());
}

void testValidationRejectsMalformedUnaryExpressions() {
    Block nullOperandBlock;
    nullOperandBlock.appendStore(
        element("A", {Index::constantValue(0)}),
        std::make_unique<Expression>(Expression::unary(UnaryOperator::Relu, nullptr)));

    DiagnosticEngine nullOperandDiagnostics;
    assert(!nullOperandBlock.validate(nullOperandDiagnostics));
    assert(nullOperandDiagnostics.hasErrors());

    auto operand = std::make_unique<Expression>(Expression::constant(1.0));
    Expression invalidOperator = Expression::unary(
        static_cast<UnaryOperator>(99), std::move(operand));
    Block invalidOperatorBlock;
    invalidOperatorBlock.appendStore(
        element("A", {Index::constantValue(0)}),
        std::make_unique<Expression>(std::move(invalidOperator)));

    DiagnosticEngine invalidOperatorDiagnostics;
    assert(!invalidOperatorBlock.validate(invalidOperatorDiagnostics));
    assert(invalidOperatorDiagnostics.hasErrors());
}

} // namespace

void runLoopIrTests() {
    testLoopStoresBoundsAndBody();
    testNestedLoopsAreOrderedInstructions();
    testExpressionsRepresentAccessesAndArithmetic();
    testBlockPreservesInstructionOrder();
    testValidationRejectsInvalidLoopAndStore();
    testIndexValidationAcceptsSupportedKinds();
    testIndexValidationRejectsMalformedKindsAndPayloads();
    testLoopUsesExclusiveUpperBound();
    testValidationRecursesThroughNestedExpressionsAndLoops();
    testValidationRejectsInvalidBinaryOperator();
    testValidationRejectsMalformedUnaryExpressions();
}
