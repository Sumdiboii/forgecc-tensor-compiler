#include "forgecc/lowering/loop_lowerer.hpp"

#include <utility>

namespace forgecc::lowering {

namespace {

using ir::Node;
using ir::OperationKind;
using ir::ValueId;

ExpressionPtr tensorRead(const std::string& tensor, std::vector<Index> indices) {
    return std::make_unique<Expression>(
        Expression::tensorElement(TensorElement{tensor, std::move(indices)}));
}

ExpressionPtr constant(double value) {
    return std::make_unique<Expression>(Expression::constant(value));
}

ExpressionPtr binary(BinaryOperator operation, ExpressionPtr left, ExpressionPtr right) {
    return std::make_unique<Expression>(
        Expression::binary(operation, std::move(left), std::move(right)));
}

ExpressionPtr relu(ExpressionPtr operand) {
    return std::make_unique<Expression>(
        Expression::unary(UnaryOperator::Relu, std::move(operand)));
}

bool sameType(const semantic::TensorType& left, const semantic::TensorType& right) {
    return left.elementType == right.elementType && left.dimensions == right.dimensions;
}

} // namespace

std::optional<LoopProgram> LoopLowerer::lower(const ir::Graph& graph) {
    diagnostics_ = common::DiagnosticEngine{};

    auto order = graph.topologicalOrder(diagnostics_);
    if (!order.has_value()) {
        return std::nullopt;
    }

    LoopProgram program;
    if (!buildTensorNames(graph, program)) {
        return std::nullopt;
    }

    for (const ir::NodeId nodeId : *order) {
        if (!lowerNode(graph, nodeId, program, program.body)) {
            return std::nullopt;
        }
    }

    if (!program.body.validate(diagnostics_)) {
        return std::nullopt;
    }
    return program;
}

const common::DiagnosticEngine& LoopLowerer::diagnostics() const {
    return diagnostics_;
}

bool LoopLowerer::buildTensorNames(const ir::Graph& graph, LoopProgram& program) {
    std::map<std::string, ValueId> names;
    for (ValueId id = 0; id < graph.valueCount(); ++id) {
        const ir::Value& value = graph.value(id);
        std::string name = value.name;
        if (name.empty()) {
            name = "value_" + std::to_string(id);
        }

        const auto [it, inserted] = names.emplace(name, id);
        if (!inserted && it->second != id) {
            reportError("graph values have duplicate loop tensor name '" + name + "'");
            return false;
        }
        program.tensorNames.emplace(id, std::move(name));
    }
    return true;
}

bool LoopLowerer::lowerNode(const ir::Graph& graph, ir::NodeId nodeId,
                            const LoopProgram& program, Block& body) {
    const Node& node = graph.node(nodeId);
    switch (node.operation) {
    case OperationKind::Relu:
        return lowerRelu(graph, node, program, body);
    case OperationKind::Matmul:
        return lowerMatmul(graph, node, program, body);
    }

    reportError("unsupported graph operation");
    return false;
}

const std::string* LoopLowerer::tensorName(const LoopProgram& program, ValueId value) const {
    const auto it = program.tensorNames.find(value);
    if (it == program.tensorNames.end()) {
        return nullptr;
    }
    return &it->second;
}

bool LoopLowerer::lowerRelu(const ir::Graph& graph, const ir::Node& node,
                            const LoopProgram& program, Block& body) {
    if (node.inputs.size() != 1) {
        reportError("relu node has incorrect input arity");
        return false;
    }

    const ir::Value& input = graph.value(node.inputs[0]);
    const ir::Value& output = graph.value(node.output);
    if (input.type.elementType != semantic::ElementType::F32 ||
        output.type.elementType != semantic::ElementType::F32) {
        reportError("relu lowering supports only f32 tensors");
        return false;
    }
    if (!sameType(input.type, output.type)) {
        reportError("relu input and output tensor types do not match");
        return false;
    }
    if (input.type.dimensions.empty()) {
        reportError("relu lowering requires a tensor with at least one dimension");
        return false;
    }

    const std::string* inputName = tensorName(program, node.inputs[0]);
    const std::string* outputName = tensorName(program, node.output);
    if (inputName == nullptr || outputName == nullptr) {
        reportError("relu node refers to an unmapped tensor value");
        return false;
    }

    Block* current = &body;
    std::vector<std::string> variables;
    variables.reserve(input.type.dimensions.size());
    for (std::size_t dimension = 0; dimension < input.type.dimensions.size(); ++dimension) {
        variables.push_back("i" + std::to_string(dimension));
        auto loop = std::make_unique<Loop>(Loop{variables.back(), 0,
                                                input.type.dimensions[dimension], 1, {}});
        current->appendLoop(std::move(loop));
        auto* nested = std::get_if<LoopPtr>(&current->instructions.back().storage);
        if (nested == nullptr || *nested == nullptr) {
            reportError("failed to construct relu loop body");
            return false;
        }
        current = &(*nested)->body;
    }

    std::vector<Index> indices;
    indices.reserve(variables.size());
    for (const std::string& variable : variables) {
        indices.push_back(Index::iteration(variable));
    }
    current->appendStore(TensorElement{*outputName, indices},
                         relu(tensorRead(*inputName, indices)));
    return true;
}

bool LoopLowerer::lowerMatmul(const ir::Graph& graph, const ir::Node& node,
                              const LoopProgram& program, Block& body) {
    if (node.inputs.size() != 2) {
        reportError("matmul node has incorrect input arity");
        return false;
    }

    const ir::Value& left = graph.value(node.inputs[0]);
    const ir::Value& right = graph.value(node.inputs[1]);
    const ir::Value& output = graph.value(node.output);
    if (left.type.elementType != semantic::ElementType::F32 ||
        right.type.elementType != semantic::ElementType::F32 ||
        output.type.elementType != semantic::ElementType::F32) {
        reportError("matmul lowering supports only f32 tensors");
        return false;
    }
    if (left.type.dimensions.size() != 2 || right.type.dimensions.size() != 2 ||
        output.type.dimensions.size() != 2) {
        reportError("matmul lowering requires rank-2 input and output tensors");
        return false;
    }
    if (left.type.dimensions[1] != right.type.dimensions[0] ||
        output.type.dimensions[0] != left.type.dimensions[0] ||
        output.type.dimensions[1] != right.type.dimensions[1]) {
        reportError("matmul tensor shapes are incompatible");
        return false;
    }

    const std::string* leftName = tensorName(program, node.inputs[0]);
    const std::string* rightName = tensorName(program, node.inputs[1]);
    const std::string* outputName = tensorName(program, node.output);
    if (leftName == nullptr || rightName == nullptr || outputName == nullptr) {
        reportError("matmul node refers to an unmapped tensor value");
        return false;
    }

    auto outer = std::make_unique<Loop>(Loop{"i", 0, left.type.dimensions[0], 1, {}});
    auto middle = std::make_unique<Loop>(Loop{"j", 0, right.type.dimensions[1], 1, {}});
    auto inner = std::make_unique<Loop>(Loop{"k", 0, left.type.dimensions[1], 1, {}});

    const std::vector<Index> outputIndices{Index::iteration("i"), Index::iteration("j")};
    middle->body.appendStore(TensorElement{*outputName, outputIndices}, constant(0.0));

    auto outputRead = tensorRead(*outputName, outputIndices);
    auto leftRead = tensorRead(
        *leftName, {Index::iteration("i"), Index::iteration("k")});
    auto rightRead = tensorRead(
        *rightName, {Index::iteration("k"), Index::iteration("j")});
    inner->body.appendStore(
        TensorElement{*outputName, outputIndices},
        binary(BinaryOperator::Add, std::move(outputRead),
               binary(BinaryOperator::Multiply, std::move(leftRead),
                      std::move(rightRead))));
    middle->body.appendLoop(std::move(inner));
    outer->body.appendLoop(std::move(middle));
    body.appendLoop(std::move(outer));
    return true;
}

void LoopLowerer::reportError(std::string message) {
    diagnostics_.reportError({1, 1}, std::move(message));
}

} // namespace forgecc::lowering
