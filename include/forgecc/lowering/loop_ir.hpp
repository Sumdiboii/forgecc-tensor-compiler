#pragma once

#include "forgecc/common/diagnostic.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace forgecc::lowering {

struct Index {
    enum class Kind {
        IterationVariable,
        Constant,
    };

    Kind kind;
    std::string variable;
    std::size_t constant = 0;

    [[nodiscard]] static Index iteration(std::string name);
    [[nodiscard]] static Index constantValue(std::size_t value);
};

struct TensorElement {
    std::string tensor;
    std::vector<Index> indices;
};

struct Expression;
using ExpressionPtr = std::unique_ptr<Expression>;

enum class BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
};

enum class UnaryOperator {
    Relu,
};

struct BinaryExpression {
    BinaryOperator operation;
    ExpressionPtr left;
    ExpressionPtr right;
};

struct UnaryExpression {
    UnaryOperator operation;
    ExpressionPtr operand;
};

struct Expression {
    using Storage = std::variant<double, TensorElement, BinaryExpression, UnaryExpression>;

    Storage storage;

    [[nodiscard]] static Expression constant(double value);
    [[nodiscard]] static Expression tensorElement(TensorElement element);
    [[nodiscard]] static Expression binary(BinaryOperator operation, ExpressionPtr left,
                                           ExpressionPtr right);
    [[nodiscard]] static Expression unary(UnaryOperator operation, ExpressionPtr operand);
};

struct StoreInstruction {
    TensorElement target;
    ExpressionPtr value;
};

struct TiledMatmulInstruction {
    std::string leftTensor;
    std::string rightTensor;
    std::string outputTensor;
    std::size_t rows;
    std::size_t shared;
    std::size_t columns;
    std::size_t tileSize;
};

struct Loop;
using LoopPtr = std::unique_ptr<Loop>;

struct Instruction {
    using Storage = std::variant<StoreInstruction, LoopPtr, TiledMatmulInstruction>;

    Storage storage;

    [[nodiscard]] static Instruction store(StoreInstruction instruction);
    [[nodiscard]] static Instruction loop(LoopPtr instruction);
    [[nodiscard]] static Instruction tiledMatmul(TiledMatmulInstruction instruction);
};

struct Block {
    std::vector<Instruction> instructions;

    void appendStore(TensorElement target, ExpressionPtr value);
    void appendLoop(LoopPtr loop);

    [[nodiscard]] bool validate(common::DiagnosticEngine& diagnostics) const;
};

struct Loop {
    // The loop uses an exclusive upper bound:
    // for (std::size_t variable = start; variable < bound; variable += step)
    std::string variable;
    std::size_t start;
    std::size_t bound;
    std::size_t step;
    Block body;

    [[nodiscard]] bool validate(common::DiagnosticEngine& diagnostics) const;
};

} // namespace forgecc::lowering
