#include "forgecc/codegen/standalone_emitter.hpp"

#include "forgecc/codegen/cpp_emitter.hpp"

#include <limits>
#include <string>
#include <utility>

namespace forgecc::codegen {

namespace {

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

std::string inputVariable(ir::ValueId valueId) {
    return "input_" + std::to_string(valueId);
}

std::string outputVariable(ir::ValueId valueId) {
    return "output_" + std::to_string(valueId);
}

} // namespace

StandaloneEmitter::StandaloneEmitter(CodegenOptions options) : options_(std::move(options)) {}

std::optional<std::string> StandaloneEmitter::generate(
    const lowering::LoopProgram& program, common::DiagnosticEngine& diagnostics) const {
    CppEmitter emitter(options_);
    auto source = emitter.generate(program, diagnostics);
    if (!source.has_value()) {
        return std::nullopt;
    }

    SourceEmitter wrapper;
    wrapper.emitLine("#include <iostream>");
    wrapper.emitBlankLine();
    wrapper.emitLine("int main() {");
    for (ir::ValueId valueId : program.inputValues) {
        const auto type = program.tensorTypes.find(valueId);
        if (type == program.tensorTypes.end() || type->second.elementType != semantic::ElementType::F32 ||
            type->second.dimensions.empty()) {
            diagnostics.reportError({1, 1}, "standalone generation found an invalid input tensor");
            return std::nullopt;
        }
        const auto count = elementCount(type->second);
        if (!count.has_value()) {
            diagnostics.reportError({1, 1}, "standalone input tensor is too large");
            return std::nullopt;
        }
        wrapper.emitLine("    " + options_.namespaceName + "::Tensor " + inputVariable(valueId) + ";");
        wrapper.emitLine("    " + inputVariable(valueId) + ".shape = " +
                         shapeInitializer(type->second) + ";");
        wrapper.emitLine("    " + inputVariable(valueId) + ".data.resize(" +
                         std::to_string(*count) + ");");
        wrapper.emitLine("    for (std::size_t index = 0; index < " + inputVariable(valueId) +
                         ".data.size(); ++index) {");
        wrapper.emitLine("        " + inputVariable(valueId) + ".data[index] = " +
                         "(index % 2 == 0 ? -1.0F : 1.0F) * " +
                         "static_cast<float>(index + 1);");
        wrapper.emitLine("    }");
    }
    for (ir::ValueId valueId : program.outputValues) {
        const auto type = program.tensorTypes.find(valueId);
        if (type == program.tensorTypes.end()) {
            diagnostics.reportError({1, 1}, "standalone generation found an invalid output tensor");
            return std::nullopt;
        }
        wrapper.emitLine("    " + options_.namespaceName + "::Tensor " + outputVariable(valueId) + ";");
    }

    std::string call = "    " + options_.namespaceName + "::compute(";
    bool first = true;
    for (ir::ValueId valueId : program.inputValues) {
        if (!first) {
            call += ", ";
        }
        call += inputVariable(valueId);
        first = false;
    }
    for (ir::ValueId valueId : program.outputValues) {
        if (!first) {
            call += ", ";
        }
        call += outputVariable(valueId);
        first = false;
    }
    call += ");";
    wrapper.emitLine(call);

    for (ir::ValueId valueId : program.outputValues) {
        const auto type = program.tensorTypes.find(valueId);
        const auto count = type == program.tensorTypes.end() ? std::nullopt
                                                              : elementCount(type->second);
        if (!count.has_value()) {
            diagnostics.reportError({1, 1}, "standalone output tensor is too large");
            return std::nullopt;
        }
        wrapper.emitLine("    for (std::size_t index = 0; index < " + outputVariable(valueId) +
                         ".data.size(); ++index) {");
        wrapper.emitLine("        std::cout << " + outputVariable(valueId) +
                         ".data[index] << '\\n';");
        wrapper.emitLine("    }");
    }
    wrapper.emitLine("    return 0;");
    wrapper.emitLine("}");

    return *source + "\n" + wrapper.str();
}

} // namespace forgecc::codegen
