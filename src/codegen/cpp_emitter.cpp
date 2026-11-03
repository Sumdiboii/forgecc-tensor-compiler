#include "forgecc/codegen/cpp_emitter.hpp"

#include <cctype>
#include <limits>
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

bool emitNode(SourceEmitter& emitter, const ir::Graph& graph, const ir::Node& node,
              common::DiagnosticEngine& diagnostics) {
    switch (node.operation) {
    case ir::OperationKind::Relu:
        return emitRelu(emitter, graph, node, diagnostics);
    case ir::OperationKind::Matmul:
        return emitMatmul(emitter, graph, node, diagnostics);
    }

    diagnostics.reportError({1, 1}, "unsupported operation in C++ code generation");
    return false;
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

} // namespace forgecc::codegen
