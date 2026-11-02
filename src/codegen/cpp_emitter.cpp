#include "forgecc/codegen/cpp_emitter.hpp"

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

std::string elementCount(const semantic::TensorType& type) {
    std::size_t count = 1;
    for (std::size_t dimension : type.dimensions) {
        count *= dimension;
    }
    return std::to_string(count);
}

void emitTensorDeclaration(SourceEmitter& emitter, const std::string& variable,
                           const semantic::TensorType& type) {
    emitter.emitLine("    Tensor " + variable + ";");
    emitter.emitLine("    " + variable + ".shape = " + shapeInitializer(type) + ";");
    emitter.emitLine("    " + variable + ".data.resize(" + elementCount(type) + ");");
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
    emitTensorDeclaration(emitter, outputName, output.type);
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
    emitTensorDeclaration(emitter, outputName, output.type);
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
