#include "forgecc/codegen/cpp_emitter.hpp"

#include "forgecc/lowering/loop_lowerer.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace forgecc::codegen {

namespace {

std::string valueVariable(const ir::Graph& graph, ir::ValueId valueId) {
    const ir::Value& value = graph.value(valueId);
    const std::string prefix = value.kind == ir::ValueKind::GraphInput ? "input_" : "value_";
    return prefix + std::to_string(valueId);
}

std::string outputVariable(ir::ValueId valueId) {
    return "output_" + std::to_string(valueId);
}

bool isF32(const semantic::TensorType& type) {
    return type.elementType == semantic::ElementType::F32;
}

std::string shapeInitializer(const semantic::TensorType& type) {
    std::string result = "{";
    for (std::size_t index = 0; index < type.dimensions.size(); ++index) {
        if (index != 0) {
            result += ", ";
        }
        result += std::to_string(type.dimensions[index]);
    }
    result += "}";
    return result;
}

std::optional<std::size_t> elementCount(const semantic::TensorType& type) {
    std::size_t count = 1;
    for (std::size_t dimension : type.dimensions) {
        if (dimension != 0 && count > std::numeric_limits<std::size_t>::max() / dimension) {
            return std::nullopt;
        }
        count *= dimension;
    }
    return count;
}

bool emitTensorDeclaration(SourceEmitter& emitter, const std::string& variable,
                           const semantic::TensorType& type,
                           common::DiagnosticEngine& diagnostics) {
    const std::optional<std::size_t> count = elementCount(type);
    if (!count.has_value()) {
        diagnostics.reportError({1, 1}, "tensor shape is too large for generated storage");
        return false;
    }

    emitter.emitLine("    Tensor " + variable + ";");
    emitter.emitLine("    " + variable + ".shape = " + shapeInitializer(type) + ";");
    emitter.emitLine("    " + variable + ".data.resize(" + std::to_string(*count) + ");");
    return true;
}

bool isIdentifier(std::string_view name) {
    if (name.empty() || (!std::isalpha(static_cast<unsigned char>(name.front())) &&
                         name.front() != '_')) {
        return false;
    }
    for (char character : name.substr(1)) {
        if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_') {
            return false;
        }
    }

    constexpr std::string_view keywords[] = {
        "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
        "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
        "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
        "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
        "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
        "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
        "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace",
        "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
        "protected", "public", "register", "reinterpret_cast", "requires", "return", "short",
        "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch",
        "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid",
        "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
        "while", "xor", "xor_eq"};
    for (std::string_view keyword : keywords) {
        if (name == keyword) {
            return false;
        }
    }
    return true;
}

bool reportUnsupportedType(const semantic::TensorType& type,
                           common::DiagnosticEngine& diagnostics) {
    if (isF32(type)) {
        return false;
    }

    diagnostics.reportError({1, 1}, "C++ code generation supports only f32 tensors");
    return true;
}

bool emitRelu(SourceEmitter& emitter, const ir::Graph& graph, const ir::Node& node,
              common::DiagnosticEngine& diagnostics) {
    const ir::Value& input = graph.value(node.inputs[0]);
    const ir::Value& output = graph.value(node.output);
    if (reportUnsupportedType(input.type, diagnostics) ||
        reportUnsupportedType(output.type, diagnostics)) {
        return false;
    }
    if (input.type.dimensions != output.type.dimensions) {
        diagnostics.reportError({1, 1}, "ReLU input and output tensor types are incompatible");
        return false;
    }

    const std::string inputName = valueVariable(graph, node.inputs[0]);
    const std::string outputName = valueVariable(graph, node.output);
    if (!emitTensorDeclaration(emitter, outputName, output.type, diagnostics)) {
        return false;
    }
    emitter.emitLine("    " + outputName + ".data.resize(" + inputName + ".data.size());");
    emitter.emitLine("    for (std::size_t index = 0; index < " + outputName +
                     ".data.size(); ++index) {");
    emitter.emitLine("        " + outputName + ".data[index] = " + inputName +
                     ".data[index] > 0.0F ? " + inputName + ".data[index] : 0.0F;");
    emitter.emitLine("    }");
    return true;
}

bool emitMatmul(SourceEmitter& emitter, const ir::Graph& graph, const ir::Node& node,
                common::DiagnosticEngine& diagnostics) {
    const ir::Value& left = graph.value(node.inputs[0]);
    const ir::Value& right = graph.value(node.inputs[1]);
    const ir::Value& output = graph.value(node.output);
    if (reportUnsupportedType(left.type, diagnostics) ||
        reportUnsupportedType(right.type, diagnostics) ||
        reportUnsupportedType(output.type, diagnostics)) {
        return false;
    }
    if (left.type.dimensions.size() != 2 || right.type.dimensions.size() != 2 ||
        output.type.dimensions.size() != 2) {
        diagnostics.reportError({1, 1}, "Matmul code generation requires rank-2 tensors");
        return false;
    }
    if (left.type.dimensions[1] != right.type.dimensions[0] ||
        output.type.dimensions[0] != left.type.dimensions[0] ||
        output.type.dimensions[1] != right.type.dimensions[1]) {
        diagnostics.reportError({1, 1}, "Matmul tensor shapes are incompatible");
        return false;
    }

    const std::string leftName = valueVariable(graph, node.inputs[0]);
    const std::string rightName = valueVariable(graph, node.inputs[1]);
    const std::string outputName = valueVariable(graph, node.output);
    const std::size_t rows = output.type.dimensions[0];
    const std::size_t inner = left.type.dimensions[1];
    const std::size_t columns = output.type.dimensions[1];
    if (!elementCount(left.type).has_value() || !elementCount(right.type).has_value()) {
        diagnostics.reportError({1, 1}, "Matmul input shape is too large for generated indexing");
        return false;
    }
    if (!emitTensorDeclaration(emitter, outputName, output.type, diagnostics)) {
        return false;
    }
    emitter.emitLine("    for (std::size_t row = 0; row < " + std::to_string(rows) +
                     "; ++row) {");
    emitter.emitLine("        for (std::size_t column = 0; column < " +
                     std::to_string(columns) + "; ++column) {");
    emitter.emitLine("            float sum = 0.0F;");
    emitter.emitLine("            for (std::size_t inner = 0; inner < " +
                     std::to_string(inner) + "; ++inner) {");
    emitter.emitLine("                sum += " + leftName + ".data[row * " +
                     std::to_string(inner) + " + inner] * " + rightName +
                     ".data[inner * " + std::to_string(columns) + " + column];");
    emitter.emitLine("            }");
    emitter.emitLine("            " + outputName + ".data[row * " +
                     std::to_string(columns) + " + column] = sum;");
    emitter.emitLine("        }");
    emitter.emitLine("    }");
    return true;
}

bool emitElementwiseBinary(SourceEmitter& emitter, const ir::Graph& graph,
                           const ir::Node& node, common::DiagnosticEngine& diagnostics) {
    const ir::Value& left = graph.value(node.inputs[0]);
    const ir::Value& right = graph.value(node.inputs[1]);
    const ir::Value& output = graph.value(node.output);
    if (reportUnsupportedType(left.type, diagnostics) ||
        reportUnsupportedType(right.type, diagnostics) ||
        reportUnsupportedType(output.type, diagnostics)) {
        return false;
    }
    if (left.type.dimensions.empty() || left.type.dimensions != right.type.dimensions ||
        left.type.dimensions != output.type.dimensions) {
        diagnostics.reportError({1, 1}, "elementwise binary tensor shapes are incompatible");
        return false;
    }

    std::string operation;
    switch (node.operation) {
    case ir::OperationKind::Add:
        operation = " + ";
        break;
    case ir::OperationKind::Subtract:
        operation = " - ";
        break;
    case ir::OperationKind::Multiply:
        operation = " * ";
        break;
    case ir::OperationKind::Divide:
        operation = " / ";
        break;
    default:
        diagnostics.reportError({1, 1}, "invalid elementwise binary operation");
        return false;
    }

    const std::string leftName = valueVariable(graph, node.inputs[0]);
    const std::string rightName = valueVariable(graph, node.inputs[1]);
    const std::string outputName = valueVariable(graph, node.output);
    if (!emitTensorDeclaration(emitter, outputName, output.type, diagnostics)) {
        return false;
    }

    emitter.emitLine("    for (std::size_t index = 0; index < " + outputName +
                     ".data.size(); ++index) {");
    emitter.emitLine("        " + outputName + ".data[index] = " + leftName +
                     ".data[index]" + operation + rightName + ".data[index];");
    emitter.emitLine("    }");
    return true;
}

bool emitNode(SourceEmitter& emitter, const ir::Graph& graph, const ir::Node& node,
              common::DiagnosticEngine& diagnostics) {
    switch (node.operation) {
    case ir::OperationKind::Relu:
        return emitRelu(emitter, graph, node, diagnostics);
    case ir::OperationKind::Matmul:
        return emitMatmul(emitter, graph, node, diagnostics);
    case ir::OperationKind::Add:
    case ir::OperationKind::Subtract:
    case ir::OperationKind::Multiply:
    case ir::OperationKind::Divide:
        return emitElementwiseBinary(emitter, graph, node, diagnostics);
    }

    diagnostics.reportError({1, 1}, "unsupported operation in C++ code generation");
    return false;
}

std::string loopValueVariable(ir::ValueId valueId, bool isInput) {
    return (isInput ? "input_" : "value_") + std::to_string(valueId);
}

struct LoopCodegenContext {
    const lowering::LoopProgram& program;
    common::DiagnosticEngine& diagnostics;
    std::set<std::string> loopVariables;
    std::set<ir::ValueId> inputValues;

    [[nodiscard]] const semantic::TensorType* tensorType(std::string_view name) const {
        for (const auto& [valueId, tensorName] : program.tensorNames) {
            if (tensorName == name) {
                const auto type = program.tensorTypes.find(valueId);
                if (type != program.tensorTypes.end()) {
                    return &type->second;
                }
            }
        }
        return nullptr;
    }

    [[nodiscard]] std::optional<ir::ValueId> tensorValue(std::string_view name) const {
        for (const auto& [valueId, tensorName] : program.tensorNames) {
            if (tensorName == name) {
                return valueId;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool isInput(ir::ValueId valueId) const {
        return inputValues.contains(valueId);
    }

    [[nodiscard]] std::string variable(ir::ValueId valueId) const {
        return loopValueVariable(valueId, isInput(valueId));
    }

    void report(std::string message) const {
        diagnostics.reportError({1, 1}, std::move(message));
    }
};

std::optional<std::string> emitLoopIndex(const lowering::Index& index,
                                         LoopCodegenContext& context) {
    switch (index.kind) {
    case lowering::Index::Kind::Constant:
        return std::to_string(index.constant);
    case lowering::Index::Kind::IterationVariable:
        if (!context.loopVariables.contains(index.variable)) {
            context.report("tensor index uses an inactive loop variable");
            return std::nullopt;
        }
        return index.variable;
    }

    context.report("invalid loop IR index kind");
    return std::nullopt;
}

std::optional<std::string> emitTensorAccess(const lowering::TensorElement& element,
                                            LoopCodegenContext& context) {
    const auto valueId = context.tensorValue(element.tensor);
    const auto* type = context.tensorType(element.tensor);
    if (!valueId.has_value() || type == nullptr) {
        context.report("loop IR refers to an unknown tensor '" + element.tensor + "'");
        return std::nullopt;
    }
    if (element.indices.size() != type->dimensions.size()) {
        context.report("tensor access has the wrong number of indices");
        return std::nullopt;
    }

    std::string offset;
    for (std::size_t index = 0; index < element.indices.size(); ++index) {
        const auto indexExpression = emitLoopIndex(element.indices[index], context);
        if (!indexExpression.has_value()) {
            return std::nullopt;
        }
        std::size_t stride = 1;
        for (std::size_t dimension = index + 1; dimension < type->dimensions.size();
             ++dimension) {
            if (type->dimensions[dimension] != 0 &&
                stride > std::numeric_limits<std::size_t>::max() /
                            type->dimensions[dimension]) {
                context.report("tensor shape is too large for generated indexing");
                return std::nullopt;
            }
            stride *= type->dimensions[dimension];
        }

        std::string term = "(" + *indexExpression + ")";
        if (stride != 1) {
            term += " * " + std::to_string(stride);
        }
        if (!offset.empty()) {
            offset += " + ";
        }
        offset += term;
    }
    return context.variable(*valueId) + ".data[" + offset + "]";
}

std::optional<std::string> emitLoopExpression(const lowering::Expression& expression,
                                              LoopCodegenContext& context) {
    if (const auto* constant = std::get_if<double>(&expression.storage)) {
        return std::to_string(*constant) + "F";
    }
    if (const auto* element = std::get_if<lowering::TensorElement>(&expression.storage)) {
        return emitTensorAccess(*element, context);
    }
    if (const auto* binary = std::get_if<lowering::BinaryExpression>(&expression.storage)) {
        if (binary->left == nullptr || binary->right == nullptr) {
            context.report("binary expression operand cannot be null");
            return std::nullopt;
        }
        const auto left = emitLoopExpression(*binary->left, context);
        const auto right = emitLoopExpression(*binary->right, context);
        if (!left.has_value() || !right.has_value()) {
            return std::nullopt;
        }
        std::string operation;
        switch (binary->operation) {
        case lowering::BinaryOperator::Add:
            operation = " + ";
            break;
        case lowering::BinaryOperator::Subtract:
            operation = " - ";
            break;
        case lowering::BinaryOperator::Multiply:
            operation = " * ";
            break;
        case lowering::BinaryOperator::Divide:
            operation = " / ";
            break;
        default:
            context.report("invalid loop IR binary operator");
            return std::nullopt;
        }
        return "(" + *left + operation + *right + ")";
    }
    if (const auto* unary = std::get_if<lowering::UnaryExpression>(&expression.storage)) {
        if (unary->operand == nullptr) {
            context.report("unary expression operand cannot be null");
            return std::nullopt;
        }
        if (unary->operation != lowering::UnaryOperator::Relu) {
            context.report("invalid loop IR unary operator");
            return std::nullopt;
        }
        const auto operand = emitLoopExpression(*unary->operand, context);
        if (!operand.has_value()) {
            return std::nullopt;
        }
        return "((" + *operand + ") > 0.0F ? (" + *operand + ") : 0.0F)";
    }

    context.report("unsupported loop IR expression");
    return std::nullopt;
}

bool emitTiledMatmul(SourceEmitter& emitter,
                     const lowering::TiledMatmulInstruction& instruction,
                     LoopCodegenContext& context) {
    const auto leftValue = context.tensorValue(instruction.leftTensor);
    const auto rightValue = context.tensorValue(instruction.rightTensor);
    const auto outputValue = context.tensorValue(instruction.outputTensor);
    const auto* leftType = context.tensorType(instruction.leftTensor);
    const auto* rightType = context.tensorType(instruction.rightTensor);
    const auto* outputType = context.tensorType(instruction.outputTensor);
    if (!leftValue.has_value() || !rightValue.has_value() || !outputValue.has_value() ||
        leftType == nullptr || rightType == nullptr || outputType == nullptr) {
        context.report("tiled matmul refers to an unknown tensor");
        return false;
    }
    if (context.isInput(*outputValue) || leftType->elementType != semantic::ElementType::F32 ||
        rightType->elementType != semantic::ElementType::F32 ||
        outputType->elementType != semantic::ElementType::F32 ||
        leftType->dimensions != std::vector<std::size_t>{instruction.rows, instruction.shared} ||
        rightType->dimensions != std::vector<std::size_t>{instruction.shared,
                                                           instruction.columns} ||
        outputType->dimensions != std::vector<std::size_t>{instruction.rows,
                                                            instruction.columns}) {
        context.report("tiled matmul tensor types or shapes are incompatible");
        return false;
    }
    if (instruction.tileSize == 0 ||
        instruction.tileSize >
            std::numeric_limits<std::size_t>::max() -
                std::max(instruction.rows,
                         std::max(instruction.shared, instruction.columns))) {
        context.report("tiled matmul tile size is invalid");
        return false;
    }
    if (!elementCount(*leftType).has_value() || !elementCount(*rightType).has_value() ||
        !elementCount(*outputType).has_value()) {
        context.report("tiled matmul tensor shape is too large");
        return false;
    }

    const std::string left = context.variable(*leftValue);
    const std::string right = context.variable(*rightValue);
    const std::string output = context.variable(*outputValue);
    const auto sizeLiteral = [](std::size_t value) {
        return "static_cast<std::size_t>(" + std::to_string(value) + ")";
    };
    const std::string tile = sizeLiteral(instruction.tileSize);
    const std::string rows = sizeLiteral(instruction.rows);
    const std::string shared = sizeLiteral(instruction.shared);
    const std::string columns = sizeLiteral(instruction.columns);

    emitter.emitLine("    for (std::size_t row = 0; row < " + rows + "; ++row) {");
    emitter.emitLine("        for (std::size_t column = 0; column < " + columns +
                     "; ++column) {");
    emitter.emitLine("            " + output + ".data[row * " + columns + " + column] = 0.0F;");
    emitter.emitLine("        }");
    emitter.emitLine("    }");
    emitter.emitLine("    for (std::size_t tile_row = 0; tile_row < " + rows +
                     "; tile_row += " + tile + ") {");
    emitter.emitLine("        for (std::size_t tile_column = 0; tile_column < " + columns +
                     "; tile_column += " + tile + ") {");
    emitter.emitLine("            for (std::size_t tile_shared = 0; tile_shared < " + shared +
                     "; tile_shared += " + tile + ") {");
    emitter.emitLine("                for (std::size_t row = tile_row; row < std::min(tile_row + " +
                     tile + ", " + rows + "); ++row) {");
    emitter.emitLine(std::string("                    for (std::size_t inner = tile_shared; inner < ") +
                     "std::min(tile_shared + " + tile + ", " + shared +"); ++inner) {");
    emitter.emitLine("                        const float a = " + left + ".data[row * " + shared +
                     " + inner];");
    emitter.emitLine(std::string("                        for (std::size_t column = tile_column; column < ") +
                     "std::min(tile_column + " + tile + ", " + columns +"); ++column) {");
    emitter.emitLine("                            " + output + ".data[row * " + columns +
                     " + column] += a * " + right + ".data[inner * " + columns +
                     " + column];");
    emitter.emitLine("                        }");
    emitter.emitLine("                    }");
    emitter.emitLine("                }");
    emitter.emitLine("            }");
    emitter.emitLine("        }");
    emitter.emitLine("    }");
    return true;
}

bool emitLoopBlock(SourceEmitter& emitter, const lowering::Block& block,
                   LoopCodegenContext& context, std::size_t indentation) {
    const std::string indent(indentation, ' ');
    for (const lowering::Instruction& instruction : block.instructions) {
        if (const auto* store = std::get_if<lowering::StoreInstruction>(&instruction.storage)) {
            const auto target = emitTensorAccess(store->target, context);
            if (!target.has_value() || store->value == nullptr) {
                if (store->value == nullptr) {
                    context.report("store expression cannot be null");
                }
                return false;
            }
            const auto output = context.tensorValue(store->target.tensor);
            if (output.has_value() && context.isInput(*output)) {
                context.report("loop IR cannot store into a graph input");
                return false;
            }
            const auto value = emitLoopExpression(*store->value, context);
            if (!value.has_value()) {
                return false;
            }
            emitter.emitLine(indent + *target + " = " + *value + ";");
            continue;
        }

        if (const auto* tiledMatmul =
                std::get_if<lowering::TiledMatmulInstruction>(&instruction.storage)) {
            if (indentation != 4) {
                context.report("tiled matmul must be emitted at the top level of a loop program");
                return false;
            }
            if (!emitTiledMatmul(emitter, *tiledMatmul, context)) {
                return false;
            }
            continue;
        }

        const auto* loop = std::get_if<lowering::LoopPtr>(&instruction.storage);
        if (loop == nullptr || *loop == nullptr || (*loop)->step == 0) {
            context.report("invalid loop instruction");
            return false;
        }
        const lowering::Loop& loopValue = **loop;
        if (loopValue.variable.empty() || context.loopVariables.contains(loopValue.variable)) {
            context.report("invalid or duplicate loop iteration variable");
            return false;
        }
        emitter.emitLine(indent + "for (std::size_t " + loopValue.variable + " = " +
                         std::to_string(loopValue.start) + "; " + loopValue.variable + " < " +
                         std::to_string(loopValue.bound) + "; " + loopValue.variable + " += " +
                         std::to_string(loopValue.step) + ") {");
        context.loopVariables.insert(loopValue.variable);
        if (!emitLoopBlock(emitter, loopValue.body, context, indentation + 4)) {
            return false;
        }
        context.loopVariables.erase(loopValue.variable);
        emitter.emitLine(indent + "}");
    }
    return true;
}

bool validateLoopMetadata(const lowering::LoopProgram& program,
                          LoopCodegenContext& context) {
    if (program.tensorNames.size() != program.tensorTypes.size()) {
        context.report("loop program tensor names and types do not match");
        return false;
    }
    std::set<std::string> names;
    for (const auto& [valueId, name] : program.tensorNames) {
        if (name.empty() || !names.insert(name).second) {
            context.report("loop program contains invalid or duplicate tensor names");
            return false;
        }
        const auto type = program.tensorTypes.find(valueId);
        if (type == program.tensorTypes.end() || !isF32(type->second) ||
            type->second.dimensions.empty() || !elementCount(type->second).has_value()) {
            context.report("loop program contains an unsupported or invalid tensor type");
            return false;
        }
    }
    for (ir::ValueId valueId : program.inputValues) {
        if (!program.tensorNames.contains(valueId) || !program.tensorTypes.contains(valueId) ||
            !context.inputValues.insert(valueId).second) {
            context.report("loop program contains an invalid input value");
            return false;
        }
    }
    for (ir::ValueId valueId : program.outputValues) {
        if (!program.tensorNames.contains(valueId) || !program.tensorTypes.contains(valueId)) {
            context.report("loop program contains an invalid output value");
            return false;
        }
    }
    return true;
}

} // namespace

void SourceEmitter::emitLine(std::string_view line) {
    source_.append(line);
    source_.push_back('\n');
}

void SourceEmitter::emitBlankLine() {
    source_.push_back('\n');
}

std::string SourceEmitter::str() const {
    return source_;
}

CppEmitter::CppEmitter(CodegenOptions options) : options_(std::move(options)) {}

std::optional<std::string> CppEmitter::generate(
    const ir::Graph& graph, common::DiagnosticEngine& diagnostics) const {
    if (!graph.validate(diagnostics)) {
        return std::nullopt;
    }
    if (!isIdentifier(options_.namespaceName)) {
        diagnostics.reportError({1, 1}, "invalid C++ namespace name in code-generation options");
        return std::nullopt;
    }

    SourceEmitter emitter;
    emitter.emitLine("// Generated by ForgeCC.");
    emitter.emitLine("#include <cstddef>");
    emitter.emitLine("#include <vector>");
    emitter.emitBlankLine();
    emitter.emitLine("namespace " + options_.namespaceName + " {");
    emitter.emitLine("struct Tensor {");
    emitter.emitLine("    std::vector<float> data;");
    emitter.emitLine("    std::vector<std::size_t> shape;");
    emitter.emitLine("};");
    emitter.emitBlankLine();

    for (ir::ValueId valueId = 0; valueId < graph.valueCount(); ++valueId) {
        const ir::Value& value = graph.value(valueId);
        if (value.kind == ir::ValueKind::GraphInput && !value.name.empty()) {
            emitter.emitLine("// input " + value.name + " -> " + valueVariable(graph, valueId));
        }
    }
    for (ir::ValueId output : graph.outputs()) {
        const ir::Value& value = graph.value(output);
        if (!value.name.empty()) {
            emitter.emitLine("// output " + value.name + " -> " + outputVariable(output));
        }
    }

    std::vector<std::string> parameters;
    for (ir::ValueId valueId = 0; valueId < graph.valueCount(); ++valueId) {
        if (graph.value(valueId).kind == ir::ValueKind::GraphInput) {
            parameters.push_back("const Tensor& " + valueVariable(graph, valueId));
        }
    }
    for (ir::ValueId output : graph.outputs()) {
        parameters.push_back("Tensor& " + outputVariable(output));
    }

    emitter.emitLine("void compute(" + [&parameters] {
        std::string result;
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            if (index != 0) {
                result += ", ";
            }
            result += parameters[index];
        }
        return result;
    }() + ") {");

    const auto order = graph.topologicalOrder(diagnostics);
    if (!order.has_value()) {
        return std::nullopt;
    }
    for (ir::NodeId nodeId : *order) {
        if (!emitNode(emitter, graph, graph.node(nodeId), diagnostics)) {
            return std::nullopt;
        }
    }
    for (ir::ValueId output : graph.outputs()) {
        emitter.emitLine("    " + outputVariable(output) + " = " +
                         valueVariable(graph, output) + ";");
    }
    emitter.emitLine("}");
    emitter.emitLine("}");
    return emitter.str();
}

std::optional<std::string> CppEmitter::generate(
    const lowering::LoopProgram& program, common::DiagnosticEngine& diagnostics) const {
    if (!isIdentifier(options_.namespaceName)) {
        diagnostics.reportError({1, 1}, "invalid C++ namespace name in code-generation options");
        return std::nullopt;
    }

    LoopCodegenContext context{program, diagnostics};
    if (!validateLoopMetadata(program, context) || !program.body.validate(diagnostics)) {
        return std::nullopt;
    }

    SourceEmitter emitter;
    emitter.emitLine("// Generated by ForgeCC.");
    emitter.emitLine("#include <algorithm>");
    emitter.emitLine("#include <cstddef>");
    emitter.emitLine("#include <vector>");
    emitter.emitBlankLine();
    emitter.emitLine("namespace " + options_.namespaceName + " {");
    emitter.emitLine("struct Tensor {");
    emitter.emitLine("    std::vector<float> data;");
    emitter.emitLine("    std::vector<std::size_t> shape;");
    emitter.emitLine("};");
    emitter.emitBlankLine();

    std::vector<std::string> parameters;
    for (ir::ValueId valueId : program.inputValues) {
        parameters.push_back("const Tensor& " + context.variable(valueId));
    }
    for (ir::ValueId valueId : program.outputValues) {
        parameters.push_back("Tensor& output_" + std::to_string(valueId));
    }
    emitter.emitLine("void compute(" + [&parameters] {
        std::string result;
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            if (index != 0) {
                result += ", ";
            }
            result += parameters[index];
        }
        return result;
    }() + ") {");

    for (const auto& [valueId, type] : program.tensorTypes) {
        if (!context.isInput(valueId)) {
            if (!emitTensorDeclaration(emitter, context.variable(valueId), type, diagnostics)) {
                return std::nullopt;
            }
        }
    }
    if (!emitLoopBlock(emitter, program.body, context, 4)) {
        return std::nullopt;
    }
    for (ir::ValueId valueId : program.outputValues) {
        emitter.emitLine("    output_" + std::to_string(valueId) + " = " +
                         context.variable(valueId) + ";");
    }
    emitter.emitLine("}");
    emitter.emitLine("}");
    return emitter.str();
}

} // namespace forgecc::codegen
