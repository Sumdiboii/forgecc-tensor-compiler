#include "forgecc/ir/graph.hpp"
#include "forgecc/optimizer/pass_manager.hpp"

#include <cassert>
#include <memory>
#include <string_view>

namespace {

using forgecc::common::DiagnosticEngine;
using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::optimizer::OptimizationPass;
using forgecc::optimizer::PassManager;
using forgecc::optimizer::PassResult;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

TensorType matrix(std::size_t rows, std::size_t columns) {
    return TensorType(ElementType::F32, {rows, columns});
}

class RenameOutputPass final : public OptimizationPass {
public:
    [[nodiscard]] std::string_view name() const override {
        return "rename-output";
    }

    [[nodiscard]] PassResult run(Graph& graph, DiagnosticEngine&) override {
        graph.setValueName(graph.outputs().front(), "optimized");
        return PassResult{true};
    }
};

class NoOpPass final : public OptimizationPass {
public:
    [[nodiscard]] std::string_view name() const override {
        return "no-op";
    }

    [[nodiscard]] PassResult run(Graph&, DiagnosticEngine&) override {
        return PassResult{false};
    }
};

class InvalidGraphPass final : public OptimizationPass {
public:
    [[nodiscard]] std::string_view name() const override {
        return "invalid-graph";
    }

    [[nodiscard]] PassResult run(Graph& graph, DiagnosticEngine&) override {
        static_cast<void>(graph.addNode(OperationKind::Relu, {99}, matrix(2, 2)));
        return PassResult{true};
    }
};

Graph graphWithOutput() {
    Graph graph;
    const auto input = graph.addInput("A", matrix(2, 2));
    const auto node = graph.addNode(OperationKind::Relu, {input}, matrix(2, 2));
    graph.addOutput(graph.node(node).output);
    return graph;
}

void testPassManagerReportsChangesAndPreservesValidity() {
    Graph graph = graphWithOutput();
    PassManager manager;
    manager.addPass(std::make_unique<NoOpPass>());
    manager.addPass(std::make_unique<RenameOutputPass>());
    DiagnosticEngine diagnostics;

    const auto result = manager.run(graph, diagnostics);
    assert(result.success);
    assert(result.changed);
    assert(graph.value(graph.outputs().front()).name == "optimized");
    assert(!diagnostics.hasErrors());
}

void testPassManagerRejectsInvalidResult() {
    Graph graph = graphWithOutput();
    PassManager manager;
    manager.addPass(std::make_unique<InvalidGraphPass>());
    DiagnosticEngine diagnostics;

    const auto result = manager.run(graph, diagnostics);
    assert(!result.success);
    assert(result.changed == false);
    assert(diagnostics.hasErrors());
}

} // namespace

void runPassManagerTests() {
    testPassManagerReportsChangesAndPreservesValidity();
    testPassManagerRejectsInvalidResult();
}
