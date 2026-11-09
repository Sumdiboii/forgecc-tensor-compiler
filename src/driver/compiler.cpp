#include "forgecc/driver/compiler.hpp"

#include "forgecc/frontend/lexer.hpp"
#include "forgecc/frontend/parser.hpp"
#include "forgecc/frontend/semantic_analyzer.hpp"
#include "forgecc/lowering/lowerer.hpp"
#include "forgecc/lowering/loop_lowerer.hpp"
#include "forgecc/optimizer/common_subexpression_elimination.hpp"
#include "forgecc/optimizer/dead_code_elimination.hpp"
#include "forgecc/optimizer/pass_manager.hpp"
#include "forgecc/codegen/standalone_emitter.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace forgecc::driver {

native::NativeCompilationResult CompilerDriver::compileGeneratedSource(
    std::string_view generatedSource, const std::filesystem::path& outputPath,
    native::NativeCompilerOptions options) const {
    return native::NativeCompiler(std::move(options)).compile(generatedSource, outputPath);
}

namespace {

std::string diagnosticsText(const common::DiagnosticEngine& diagnostics) {
    std::string result;
    for (const common::Diagnostic& diagnostic : diagnostics.diagnostics()) {
        result += std::to_string(diagnostic.location.line) + ":" +
                  std::to_string(diagnostic.location.column) + ": " + diagnostic.message + "\n";
    }
    return result;
}

CompilationResult failure(std::string message) {
    return CompilationResult{false, {}, std::move(message), {}, -1};
}

} // namespace

CompilationResult CompilerDriver::compileAndRun(
    const std::filesystem::path& outputPath, native::NativeCompilerOptions nativeOptions,
    codegen::CodegenOptions codegenOptions) const {
    if (source_.contents.empty()) {
        return failure("source program is empty");
    }

    frontend::Lexer lexer(source_.contents);
    std::vector<frontend::Token> tokens = lexer.tokenize();
    if (lexer.diagnostics().hasErrors()) {
        return failure("lexical analysis failed:\n" + diagnosticsText(lexer.diagnostics()));
    }

    frontend::Parser parser(std::move(tokens));
    std::unique_ptr<frontend::Program> program = parser.parseProgram();
    if (program == nullptr || parser.diagnostics().hasErrors()) {
        return failure("parsing failed:\n" + diagnosticsText(parser.diagnostics()));
    }
    if (program->statements.empty()) {
        return failure("source program is empty");
    }

    frontend::SemanticAnalyzer analyzer;
    if (!analyzer.analyze(*program)) {
        return failure("semantic analysis failed:\n" + diagnosticsText(analyzer.diagnostics()));
    }

    lowering::GraphLowerer graphLowerer;
    auto graph = graphLowerer.lower(*program, analyzer);
    if (!graph.has_value()) {
        return failure("graph lowering failed:\n" + diagnosticsText(graphLowerer.diagnostics()));
    }
    common::DiagnosticEngine graphDiagnostics;
    if (!graph->validate(graphDiagnostics)) {
        return failure("graph validation failed:\n" + diagnosticsText(graphDiagnostics));
    }

    optimizer::PassManager passManager;
    passManager.addPass(std::make_unique<optimizer::CommonSubexpressionElimination>());
    passManager.addPass(std::make_unique<optimizer::DeadCodeElimination>());
    common::DiagnosticEngine optimizationDiagnostics;
    const optimizer::PassManagerResult optimization =
        passManager.run(*graph, optimizationDiagnostics);
    if (!optimization.success) {
        return failure("graph optimization failed:\n" +
                       diagnosticsText(optimizationDiagnostics));
    }

    lowering::LoopLowerer loopLowerer;
    auto loopProgram = loopLowerer.lower(*graph);
    if (!loopProgram.has_value()) {
        return failure("loop lowering failed:\n" + diagnosticsText(loopLowerer.diagnostics()));
    }

    codegen::StandaloneEmitter standaloneEmitter(std::move(codegenOptions));
    common::DiagnosticEngine codegenDiagnostics;
    auto generatedSource = standaloneEmitter.generate(*loopProgram, codegenDiagnostics);
    if (!generatedSource.has_value()) {
        return failure("code generation failed:\n" + diagnosticsText(codegenDiagnostics));
    }

    const native::NativeCompilationResult compilation =
        native::NativeCompiler(std::move(nativeOptions)).compile(*generatedSource, outputPath);
    if (!compilation.success) {
        return {false, compilation.outputPath, "native compilation failed:\n" +
                                                   compilation.diagnostics,
                {}, compilation.exitCode};
    }

    const native::NativeExecutionResult execution =
        native::NativeExecutor{}.execute(compilation.outputPath);
    if (!execution.success) {
        return {false, compilation.outputPath, "native execution failed:\n" + execution.output,
                execution.output, execution.exitCode};
    }

    return {true, compilation.outputPath, {}, execution.output, execution.exitCode};
}

} // namespace forgecc::driver
