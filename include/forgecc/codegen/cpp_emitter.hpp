#pragma once

#include "forgecc/codegen/codegen_options.hpp"
#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/graph.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace forgecc::codegen {

class SourceEmitter {
public:
    void emitLine(std::string_view line);
    void emitBlankLine();

    [[nodiscard]] std::string str() const;

private:
    std::string source_;
};

class CppEmitter {
public:
    explicit CppEmitter(CodegenOptions options = {});

    [[nodiscard]] std::optional<std::string> generate(
        const ir::Graph& graph, common::DiagnosticEngine& diagnostics) const;

private:
    CodegenOptions options_;
};

} // namespace forgecc::codegen
