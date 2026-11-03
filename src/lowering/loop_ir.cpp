#include "forgecc/lowering/loop_ir.hpp"

#include <utility>

namespace forgecc::lowering {

namespace {

void report(common::DiagnosticEngine& diagnostics, std::string message) {
    diagnostics.reportError({1, 1}, std::move(message));
}

bool validateIndex(const Index& index, common::DiagnosticEngine& diagnostics) {
    switch (index.kind) {
    case Index::Kind::IterationVariable:
        if (index.variable.empty()) {
            report(diagnostics, "iteration index variable cannot be empty");
            return false;
        }
        if (index.constant != 0) {
            report(diagnostics,
                   "iteration-variable index cannot contain a constant value");
            return false;
        }
        return true;
    case Index::Kind::Constant:
        if (!index.variable.empty()) {
            report(diagnostics, "constant index cannot contain an iteration variable");
            return false;
        }
        return true;
    }

    report(diagnostics, "invalid tensor index kind");
    return false;
}

bool validateBinaryOperator(BinaryOperator operation,
                            common::DiagnosticEngine& diagnostics) {
    switch (operation) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
    case BinaryOperator::Multiply:
    case BinaryOperator::Divide:
        return true;
    }

    report(diagnostics, "invalid loop IR binary operator");
    return false;
}

bool validateElement(const TensorElement& element, common::DiagnosticEngine& diagnostics) {
    bool valid = true;
    if (element.tensor.empty()) {
        report(diagnostics, "tensor name cannot be empty");
        valid = false;
    }
    if (element.indices.empty()) {
        report(diagnostics, "tensor element access requires at least one index");
        valid = false;
    }
    for (const Index& index : element.indices) {
        valid = validateIndex(index, diagnostics) && valid;
    }
    return valid;
}

bool validateExpression(const Expression& expression, common::DiagnosticEngine& diagnostics) {
    if (const auto* element = std::get_if<TensorElement>(&expression.storage)) {
        return validateElement(*element, diagnostics);
    }
    if (const auto* binary = std::get_if<BinaryExpression>(&expression.storage)) {
        bool valid = true;
        if (!validateBinaryOperator(binary->operation, diagnostics)) {
            valid = false;
        }
        if (binary->left == nullptr || binary->right == nullptr) {
            report(diagnostics, "binary expression operands cannot be null");
            valid = false;
        }
        if (binary->left != nullptr) {
            valid = validateExpression(*binary->left, diagnostics) && valid;
        }
        if (binary->right != nullptr) {
            valid = validateExpression(*binary->right, diagnostics) && valid;
        }
        return valid;
    }
    return true;
}

} // namespace

Index Index::iteration(std::string name) {
    return Index{Kind::IterationVariable, std::move(name), 0};
}

Index Index::constantValue(std::size_t value) {
    return Index{Kind::Constant, {}, value};
}

Expression Expression::constant(double value) {
    return Expression{value};
}

Expression Expression::tensorElement(TensorElement element) {
    return Expression{std::move(element)};
}

Expression Expression::binary(BinaryOperator operation, ExpressionPtr left,
                              ExpressionPtr right) {
    return Expression{BinaryExpression{operation, std::move(left), std::move(right)}};
}

Instruction Instruction::store(StoreInstruction instruction) {
    return Instruction{std::move(instruction)};
}

Instruction Instruction::loop(LoopPtr instruction) {
    return Instruction{std::move(instruction)};
}

void Block::appendStore(TensorElement target, ExpressionPtr value) {
    instructions.push_back(Instruction::store(StoreInstruction{std::move(target), std::move(value)}));
}

void Block::appendLoop(LoopPtr loop) {
    instructions.push_back(Instruction::loop(std::move(loop)));
}

bool Block::validate(common::DiagnosticEngine& diagnostics) const {
    bool valid = true;
    for (const Instruction& instruction : instructions) {
        if (const auto* store = std::get_if<StoreInstruction>(&instruction.storage)) {
            if (!validateElement(store->target, diagnostics)) {
                valid = false;
            }
            if (store->value == nullptr) {
                report(diagnostics, "store expression cannot be null");
                valid = false;
            } else if (!validateExpression(*store->value, diagnostics)) {
                valid = false;
            }
        } else if (const auto* loop = std::get_if<LoopPtr>(&instruction.storage)) {
            if (*loop == nullptr || !(*loop)->validate(diagnostics)) {
                valid = false;
            }
        }
    }
    return valid;
}

bool Loop::validate(common::DiagnosticEngine& diagnostics) const {
    bool valid = true;
    if (variable.empty()) {
        report(diagnostics, "loop iteration variable cannot be empty");
        valid = false;
    }
    if (step == 0) {
        report(diagnostics, "loop step must be greater than zero");
        valid = false;
    }
    if (!body.validate(diagnostics)) {
        valid = false;
    }
    return valid;
}

} // namespace forgecc::lowering
