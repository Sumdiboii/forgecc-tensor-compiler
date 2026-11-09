#pragma once

#include "forgecc/codegen/codegen_options.hpp"
#include "forgecc/common/diagnostic.hpp"
#include "forgecc/lowering/loop_lowerer.hpp"

#include <optional>
#include <string>

namespace forgecc::codegen {

class StandaloneEmitter {
public:
    explicit StandaloneEmitter(CodegenOptions options = {});

    [[nodiscard]] std::optional<std::string> generate(
        const lowering::LoopProgram& program, common::DiagnosticEngine& diagnostics) const;

private:
    CodegenOptions options_;
};

} // namespace forgecc::codegen
