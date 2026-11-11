#pragma once

#include "forgecc/common/diagnostic.hpp"
#include "forgecc/ir/graph.hpp"
#include "forgecc/lowering/loop_ir.hpp"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace forgecc::lowering {

struct LoopProgram {
    Block body;
    std::map<ir::ValueId, std::string> tensorNames;
    std::map<ir::ValueId, semantic::TensorType> tensorTypes;
    std::vector<ir::ValueId> inputValues;
    std::vector<ir::ValueId> outputValues;
};

struct LoopLowererOptions {
    std::optional<std::size_t> matmulTileSize;
};

class LoopLowerer {
public:
    explicit LoopLowerer(LoopLowererOptions options = {}) : options_(std::move(options)) {}

    [[nodiscard]] std::optional<LoopProgram> lower(const ir::Graph& graph);
    [[nodiscard]] const common::DiagnosticEngine& diagnostics() const;

private:
    [[nodiscard]] bool buildTensorNames(const ir::Graph& graph, LoopProgram& program);
    [[nodiscard]] bool lowerNode(const ir::Graph& graph, ir::NodeId nodeId,
                                 const LoopProgram& program, Block& body);
    [[nodiscard]] bool lowerRelu(const ir::Graph& graph, const ir::Node& node,
                                 const LoopProgram& program, Block& body);
    [[nodiscard]] bool lowerMatmul(const ir::Graph& graph, const ir::Node& node,
                                   const LoopProgram& program, Block& body);
    [[nodiscard]] bool lowerTiledMatmul(const ir::Graph& graph, const ir::Node& node,
                                        const LoopProgram& program, Block& body);
    [[nodiscard]] bool lowerElementwiseBinary(const ir::Graph& graph, const ir::Node& node,
                                              const LoopProgram& program, Block& body);
    [[nodiscard]] const std::string* tensorName(const LoopProgram& program,
                                                ir::ValueId value) const;
    void reportError(std::string message);

    LoopLowererOptions options_;
    common::DiagnosticEngine diagnostics_;
};

} // namespace forgecc::lowering
