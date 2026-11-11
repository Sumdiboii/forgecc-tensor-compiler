#include "forgecc/codegen/cpp_emitter.hpp"
#include "forgecc/ir/graph.hpp"
#include "forgecc/lowering/loop_lowerer.hpp"
#include "forgecc/native/native_compiler.hpp"
#include "forgecc/native/native_executor.hpp"
#include "forgecc/optimizer/common_subexpression_elimination.hpp"
#include "forgecc/optimizer/dead_code_elimination.hpp"
#include "forgecc/optimizer/pass_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using forgecc::common::DiagnosticEngine;
using forgecc::ir::Graph;
using forgecc::ir::OperationKind;
using forgecc::ir::ValueId;
using forgecc::semantic::ElementType;
using forgecc::semantic::TensorType;

struct Options {
    std::size_t warmupIterations = 5;
    std::size_t measuredIterations = 30;
    std::filesystem::path outputPath = "benchmarks/results/cpu_benchmarks.csv";
};

enum class WorkloadKind {
    Add,
    Relu,
    Matmul,
};

struct Workload {
    std::string name;
    WorkloadKind kind;
    std::vector<std::size_t> leftShape;
    std::vector<std::size_t> rightShape;
};

struct LoopVariant {
    std::string implementation;
    std::string namespaceName;
    std::string source;
};

struct SampleStatistics {
    double averageNanoseconds = 0.0;
    double medianNanoseconds = 0.0;
    double minimumNanoseconds = 0.0;
    double maximumNanoseconds = 0.0;
};

struct WorkloadResult {
    std::string workload;
    std::string dimensions;
    std::map<std::string, SampleStatistics> statistics;
    std::map<std::string, std::vector<double>> samples;
    std::map<std::string, double> checksums;
    std::size_t warmupIterations = 0;
    std::size_t measuredIterations = 0;
    double compilationMilliseconds = 0.0;
    std::size_t sourceBytes = 0;
    std::uintmax_t executableBytes = 0;
};

TensorType f32(std::vector<std::size_t> dimensions) {
    return TensorType(ElementType::F32, std::move(dimensions));
}

std::optional<std::size_t> elementCount(const TensorType& type) {
    std::size_t count = 1;
    for (const std::size_t dimension : type.dimensions) {
        if (dimension != 0 && count > std::numeric_limits<std::size_t>::max() / dimension) {
            return std::nullopt;
        }
        count *= dimension;
    }
    return count;
}

std::string shapeInitializer(const std::vector<std::size_t>& dimensions) {
    std::string result = "{";
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            result += ", ";
        }
        result += std::to_string(dimensions[index]);
    }
    result += "}";
    return result;
}

std::string dimensionsText(const Workload& workload) {
    std::string result = "[";
    for (std::size_t index = 0; index < workload.leftShape.size(); ++index) {
        if (index != 0) {
            result += "x";
        }
        result += std::to_string(workload.leftShape[index]);
    }
    if (workload.kind == WorkloadKind::Matmul) {
        result += "][";
        for (std::size_t index = 0; index < workload.rightShape.size(); ++index) {
            if (index != 0) {
                result += "x";
            }
            result += std::to_string(workload.rightShape[index]);
        }
    }
    return result + "]";
}

Graph makeGraph(const Workload& workload) {
    Graph graph;
    const ValueId left = graph.addInput("A", f32(workload.leftShape));
    std::vector<ValueId> inputs{left};
    TensorType outputType = f32(workload.leftShape);
    if (workload.kind == WorkloadKind::Relu) {
        const auto node = graph.addNode(OperationKind::Relu, inputs, outputType);
        graph.setValueName(graph.node(node).output, "out");
        graph.addOutput(graph.node(node).output);
        return graph;
    }

    const ValueId right = graph.addInput("B", f32(workload.rightShape));
    inputs.push_back(right);
    if (workload.kind == WorkloadKind::Matmul) {
        outputType = f32({workload.leftShape[0], workload.rightShape[1]});
    }
    const OperationKind operation = workload.kind == WorkloadKind::Add
                                        ? OperationKind::Add
                                        : OperationKind::Matmul;
    const auto node = graph.addNode(operation, std::move(inputs), outputType);
    graph.setValueName(graph.node(node).output, "out");
    graph.addOutput(graph.node(node).output);
    return graph;
}

std::string baselineFunction(const Workload& workload, const TensorType& outputType,
                             std::size_t outputId) {
    const auto count = elementCount(outputType);
    if (!count.has_value()) {
        return {};
    }

    std::string source;
    source += "void baseline_compute(const forgecc_optimized::Tensor& input_0";
    if (workload.kind != WorkloadKind::Relu) {
        source += ", const forgecc_optimized::Tensor& input_1";
    }
    source += ", forgecc_optimized::Tensor& output_" + std::to_string(outputId) + ") {\n";
    source += "    output_" + std::to_string(outputId) + ".shape = " +
              shapeInitializer(outputType.dimensions) + ";\n";
    source += "    output_" + std::to_string(outputId) + ".data.resize(" +
              std::to_string(*count) + ");\n";

    if (workload.kind == WorkloadKind::Matmul) {
        const std::size_t rows = workload.leftShape[0];
        const std::size_t inner = workload.leftShape[1];
        const std::size_t columns = workload.rightShape[1];
        source += "    for (std::size_t row = 0; row < " + std::to_string(rows) + "; ++row) {\n";
        source += "        for (std::size_t column = 0; column < " +
                  std::to_string(columns) + "; ++column) {\n";
        source += "            float sum = 0.0F;\n";
        source += "            for (std::size_t shared = 0; shared < " +
                  std::to_string(inner) + "; ++shared) {\n";
        source += "                sum += input_0.data[row * " + std::to_string(inner) +
                  " + shared] * input_1.data[shared * " + std::to_string(columns) +
                  " + column];\n";
        source += "            }\n";
        source += "            output_" + std::to_string(outputId) + ".data[row * " +
                  std::to_string(columns) + " + column] = sum;\n";
        source += "        }\n    }\n";
    } else {
        source += "    for (std::size_t index = 0; index < output_" +
                  std::to_string(outputId) + ".data.size(); ++index) {\n";
        source += "        output_" + std::to_string(outputId) + ".data[index] = ";
        if (workload.kind == WorkloadKind::Relu) {
            source += "input_0.data[index] > 0.0F ? input_0.data[index] : 0.0F";
        } else {
            source += "input_0.data[index] + input_1.data[index]";
        }
        source += ";\n    }\n";
    }
    source += "}\n\n";
    return source;
}

std::string inputInitialization(std::string_view namespaceName, std::string_view variable,
                                const TensorType& type) {
    const auto count = elementCount(type);
    if (!count.has_value()) {
        return {};
    }
    std::string source = "    " + std::string(namespaceName) + "::Tensor " +
                         std::string(variable) + ";\n";
    source += "    " + std::string(variable) + ".shape = " +
              shapeInitializer(type.dimensions) + ";\n";
    source += "    " + std::string(variable) + ".data.resize(" + std::to_string(*count) +");\n";
    source += "    for (std::size_t index = 0; index < " + std::string(variable) +
              ".data.size(); ++index) {\n";
    source += "        " + std::string(variable) + ".data[index] = " +
              "(index % 2 == 0 ? -1.0F : 1.0F) * static_cast<float>(index + 1);\n";
    source += "    }\n";
    return source;
}

std::string timedBlock(std::string_view implementation, std::string_view function,
                       std::string_view namespaceName, std::string_view tensorNamespace,
                       const Workload& workload,
                       const TensorType& leftType, const TensorType& rightType,
                       std::size_t outputId, std::size_t warmup, std::size_t iterations) {
    const std::string prefix = std::string(namespaceName) + "_";
    std::string source = "  {\n";
    source += inputInitialization(tensorNamespace, prefix + "input_0", leftType);
    if (workload.kind != WorkloadKind::Relu) {
        source += inputInitialization(tensorNamespace, prefix + "input_1", rightType);
    }
    source += "    " + std::string(tensorNamespace) + "::Tensor " + prefix + "output_" +
              std::to_string(outputId) + ";\n";
    source += "    for (std::size_t warmup = 0; warmup < " + std::to_string(warmup) +
              "; ++warmup) {\n";
    source += "      " + std::string(function) + "(" + prefix + "input_0";
    if (workload.kind != WorkloadKind::Relu) {
        source += ", " + prefix + "input_1";
    }
    source += ", " + prefix + "output_" + std::to_string(outputId) + ");\n    }\n";
    source += "    for (std::size_t iteration = 0; iteration < " +
              std::to_string(iterations) + "; ++iteration) {\n";
    source += "      const auto start = std::chrono::steady_clock::now();\n";
    source += "      " + std::string(function) + "(" + prefix + "input_0";
    if (workload.kind != WorkloadKind::Relu) {
        source += ", " + prefix + "input_1";
    }
    source += ", " + prefix + "output_" + std::to_string(outputId) + ");\n";
    source += "      const auto end = std::chrono::steady_clock::now();\n";
    source += "      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();\n";
    source += "      std::cout << \"RESULT," + std::string(implementation) +
              ",\" << iteration << \",\" << elapsed << \"\\n\";\n";
    source += "    }\n";
    source += "    float checksum = 0.0F;\n";
    source += "    for (const float value : " + prefix + "output_" +
              std::to_string(outputId) + ".data) {\n";
    source += "      checksum += value;\n    }\n";
    source += "    volatile float consumedChecksum = checksum;\n";
    source += "    std::cout << \"CHECKSUM," + std::string(implementation) +
              ",\" << consumedChecksum << \"\\n\";\n  }\n";
    return source;
}

std::string makeHarness(const Workload& workload, const Graph& graph,
                        std::string graphSource, std::string loopSource,
                        const std::vector<LoopVariant>& tiledVariants,
                        std::size_t warmup, std::size_t iterations) {
    const ValueId outputId = graph.outputs().front();
    const TensorType& leftType = graph.value(0).type;
    const TensorType& rightType = workload.kind == WorkloadKind::Relu
                                      ? leftType
                                      : graph.value(1).type;
    const TensorType& outputType = graph.value(outputId).type;
    const std::string baseline = baselineFunction(workload, outputType, outputId);

    std::string source = std::move(graphSource) + "\n" + std::move(loopSource) + "\n";
    for (const LoopVariant& variant : tiledVariants) {
        source += variant.source + "\n";
    }
    source += "#include <chrono>\n#include <iostream>\n\n";
    source += baseline;
    source += "int main() {\n";
    source += timedBlock("baseline", "baseline_compute", "baseline", "forgecc_optimized",
                         workload, leftType,
                         rightType, outputId, warmup, iterations);
    source += timedBlock("optimized", "forgecc_optimized::compute", "optimized",
                         "forgecc_optimized", workload,
                         leftType, rightType, outputId, warmup, iterations);
    source += timedBlock("forgecc", "forgecc_loop::compute", "loop", "forgecc_loop", workload,
                         leftType,
                         rightType, outputId, warmup, iterations);
    for (const LoopVariant& variant : tiledVariants) {
        source += timedBlock(variant.implementation, variant.namespaceName + "::compute",
                             variant.namespaceName, variant.namespaceName, workload, leftType,
                             rightType, outputId, warmup, iterations);
    }
    source += "  return 0;\n}\n";
    return source;
}

struct RawMeasurements {
    std::map<std::string, std::vector<double>> samples;
    std::map<std::string, double> checksums;
};

std::optional<RawMeasurements> parseMeasurements(const std::string& output,
                                                  std::size_t expectedIterations,
                                                  const std::vector<std::string>& implementations) {
    RawMeasurements measurements;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string kind;
        std::string implementation;
        std::string value;
        std::getline(fields, kind, ',');
        std::getline(fields, implementation, ',');
        std::getline(fields, value, ',');
        if (kind == "RESULT") {
            std::string duration;
            std::getline(fields, duration, ',');
            try {
                if (value.empty() || duration.empty()) {
                    return std::nullopt;
                }
                const double nanoseconds = std::stod(duration);
                if (nanoseconds < 0.0) {
                    return std::nullopt;
                }
                measurements.samples[implementation].push_back(nanoseconds);
            } catch (const std::exception&) {
                return std::nullopt;
            }
        } else if (kind == "CHECKSUM") {
            try {
                measurements.checksums[implementation] = std::stod(value);
            } catch (const std::exception&) {
                return std::nullopt;
            }
        } else if (!line.empty()) {
            return std::nullopt;
        }
    }

    for (const std::string& implementation : implementations) {
        if (measurements.samples[implementation].size() != expectedIterations ||
            !measurements.checksums.contains(implementation)) {
            return std::nullopt;
        }
    }
    return measurements;
}

SampleStatistics statistics(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    SampleStatistics result;
    result.minimumNanoseconds = samples.front();
    result.maximumNanoseconds = samples.back();
    for (const double sample : samples) {
        result.averageNanoseconds += sample;
    }
    result.averageNanoseconds /= static_cast<double>(samples.size());
    const std::size_t middle = samples.size() / 2;
    result.medianNanoseconds = samples.size() % 2 == 0
                                   ? (samples[middle - 1] + samples[middle]) / 2.0
                                   : samples[middle];
    return result;
}

std::vector<Workload> workloads() {
    return {
        {"add-1024", WorkloadKind::Add, {1024}, {1024}},
        {"add-4096", WorkloadKind::Add, {4096}, {4096}},
        {"relu-1024", WorkloadKind::Relu, {1024}, {}},
        {"relu-4096", WorkloadKind::Relu, {4096}, {}},
        {"matmul-2x3-3x2", WorkloadKind::Matmul, {2, 3}, {3, 2}},
        {"matmul-3x2-2x4", WorkloadKind::Matmul, {3, 2}, {2, 4}},
        {"matmul-1x3-3x1", WorkloadKind::Matmul, {1, 3}, {3, 1}},
        {"matmul-4x3-3x5", WorkloadKind::Matmul, {4, 3}, {3, 5}},
        {"matmul-8", WorkloadKind::Matmul, {8, 8}, {8, 8}},
        {"matmul-16", WorkloadKind::Matmul, {16, 16}, {16, 16}},
        {"matmul-32", WorkloadKind::Matmul, {32, 32}, {32, 32}},
        {"matmul-64", WorkloadKind::Matmul, {64, 64}, {64, 64}},
        {"matmul-128", WorkloadKind::Matmul, {128, 128}, {128, 128}},
        {"matmul-64x96-96x80", WorkloadKind::Matmul, {64, 96}, {96, 80}},
    };
}

std::optional<Options> parseOptions(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if ((argument == "--warmup" || argument == "--iterations" || argument == "--output") &&
            index + 1 >= argc) {
            return std::nullopt;
        }
        if (argument == "--warmup") {
            try {
                options.warmupIterations = std::stoull(argv[++index]);
            } catch (const std::exception&) {
                return std::nullopt;
            }
        } else if (argument == "--iterations") {
            try {
                options.measuredIterations = std::stoull(argv[++index]);
            } catch (const std::exception&) {
                return std::nullopt;
            }
        } else if (argument == "--output") {
            options.outputPath = argv[++index];
        } else {
            return std::nullopt;
        }
    }
    if (options.measuredIterations == 0) {
        return std::nullopt;
    }
    return options;
}

std::optional<WorkloadResult> runWorkload(const Workload& workload, const Options& options,
                                          std::size_t workloadIndex) {
    Graph graph = makeGraph(workload);
    DiagnosticEngine diagnostics;
    Graph optimizedGraph = graph;
    forgecc::optimizer::PassManager passManager;
    passManager.addPass(std::make_unique<forgecc::optimizer::CommonSubexpressionElimination>());
    passManager.addPass(std::make_unique<forgecc::optimizer::DeadCodeElimination>());
    if (!passManager.run(optimizedGraph, diagnostics).success) {
        std::cerr << "benchmark graph optimization failed: " << workload.name << '\n';
        return std::nullopt;
    }

    forgecc::codegen::CppEmitter graphEmitter({"forgecc_optimized"});
    auto graphSource = graphEmitter.generate(optimizedGraph, diagnostics);
    if (!graphSource.has_value()) {
        std::cerr << "benchmark graph code generation failed: " << workload.name << '\n';
        return std::nullopt;
    }
    forgecc::lowering::LoopLowerer loopLowerer;
    auto loopProgram = loopLowerer.lower(optimizedGraph);
    if (!loopProgram.has_value()) {
        std::cerr << "benchmark loop lowering failed: " << workload.name << '\n';
        return std::nullopt;
    }
    forgecc::codegen::CppEmitter loopEmitter({"forgecc_loop"});
    auto loopSource = loopEmitter.generate(*loopProgram, diagnostics);
    if (!loopSource.has_value()) {
        std::cerr << "benchmark loop code generation failed: " << workload.name << '\n';
        return std::nullopt;
    }

    std::vector<LoopVariant> tiledVariants;
    const std::vector<std::size_t> tileSizes = workload.kind == WorkloadKind::Matmul
                                                   ? std::vector<std::size_t>{8, 16, 32}
                                                   : std::vector<std::size_t>{};
    for (const std::size_t tileSize : tileSizes) {
        const std::string namespaceName = "forgecc_tiled_" + std::to_string(tileSize);
        forgecc::lowering::LoopLowerer tiledLowerer(
            forgecc::lowering::LoopLowererOptions{tileSize});
        auto tiledProgram = tiledLowerer.lower(optimizedGraph);
        if (!tiledProgram.has_value()) {
            std::cerr << "benchmark tiled loop lowering failed: " << workload.name
                      << " tile=" << tileSize << '\n';
            return std::nullopt;
        }
        forgecc::codegen::CppEmitter tiledEmitter({namespaceName});
        auto tiledSource = tiledEmitter.generate(*tiledProgram, diagnostics);
        if (!tiledSource.has_value()) {
            std::cerr << "benchmark tiled loop code generation failed: " << workload.name
                      << " tile=" << tileSize << '\n';
            return std::nullopt;
        }
        tiledVariants.push_back(
            {"tiled-" + std::to_string(tileSize), namespaceName, std::move(*tiledSource)});
    }

    const std::string source = makeHarness(workload, optimizedGraph,
                                           std::move(*graphSource), std::move(*loopSource),
                                           tiledVariants,
                                           options.warmupIterations, options.measuredIterations);
    const auto outputPath = std::filesystem::current_path() /
                            (".forgecc-benchmark-" + std::to_string(getpid()) + "-" +
                             std::to_string(workloadIndex));
    std::error_code filesystemError;
    std::filesystem::remove(outputPath, filesystemError);
    const auto compilationStart = std::chrono::steady_clock::now();
    const auto compilation = forgecc::native::NativeCompiler{}.compile(source, outputPath);
    const auto compilationEnd = std::chrono::steady_clock::now();
    if (!compilation.success) {
        std::cerr << compilation.diagnostics;
        std::filesystem::remove(outputPath, filesystemError);
        return std::nullopt;
    }

    std::error_code sizeError;
    const std::uintmax_t executableBytes = std::filesystem::file_size(outputPath, sizeError);
    const auto execution = forgecc::native::NativeExecutor{}.execute(outputPath);
    std::filesystem::remove(outputPath, filesystemError);
    if (!execution.success) {
        std::cerr << execution.output;
        return std::nullopt;
    }
    std::vector<std::string> implementations{"baseline", "optimized", "forgecc"};
    for (const LoopVariant& variant : tiledVariants) {
        implementations.push_back(variant.implementation);
    }
    const auto measurements =
        parseMeasurements(execution.output, options.measuredIterations, implementations);
    if (!measurements.has_value()) {
        std::cerr << "benchmark output was malformed: " << workload.name << '\n';
        std::cerr << execution.output;
        return std::nullopt;
    }
    const double referenceChecksum = measurements->checksums.at("baseline");
    for (const auto& [implementation, checksum] : measurements->checksums) {
        if (std::abs(checksum - referenceChecksum) > 1.0e-5) {
            std::cerr << "benchmark implementations disagree: " << workload.name << '\n';
            return std::nullopt;
        }
    }

    WorkloadResult result;
    result.workload = workload.name;
    result.dimensions = dimensionsText(workload);
    result.warmupIterations = options.warmupIterations;
    result.measuredIterations = options.measuredIterations;
    result.compilationMilliseconds =
        std::chrono::duration<double, std::milli>(compilationEnd - compilationStart).count();
    result.sourceBytes = source.size();
    result.executableBytes = sizeError ? 0 : executableBytes;
    for (const auto& [implementation, samples] : measurements->samples) {
        result.statistics.emplace(implementation, statistics(samples));
    }
    result.samples = measurements->samples;
    result.checksums = measurements->checksums;
    return result;
}

bool writeResults(const std::filesystem::path& path,
                  const std::vector<WorkloadResult>& results) {
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return false;
        }
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "workload,implementation,dimensions,warmup_iterations,measurement_iterations,"
              "sample_index,sample_ns,average_ns,median_ns,min_ns,max_ns,compile_ms,source_bytes,"
              "executable_bytes,speedup_vs_baseline\n";
    for (const WorkloadResult& result : results) {
        const double baseline = result.statistics.at("baseline").medianNanoseconds;
        for (const auto& [implementation, stats] : result.statistics) {
            const auto& samples = result.samples.at(implementation);
            for (std::size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
                output << result.workload << "," << implementation << ",\""
                       << result.dimensions << "\"," << result.warmupIterations << ","
                       << result.measuredIterations << "," << sampleIndex << ","
                       << std::fixed << std::setprecision(3) << samples[sampleIndex] << ","
                       << stats.averageNanoseconds << "," << stats.medianNanoseconds << ","
                       << stats.minimumNanoseconds << "," << stats.maximumNanoseconds << ","
                       << result.compilationMilliseconds << "," << result.sourceBytes << ","
                       << result.executableBytes << "," << baseline / stats.medianNanoseconds
                       << "\n";
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    const auto options = parseOptions(argc, argv);
    if (!options.has_value()) {
        std::cerr << "Usage: forgecc_benchmarks [--warmup N] [--iterations N] [--output FILE]\n";
        return 2;
    }

    std::vector<WorkloadResult> results;
    const auto cases = workloads();
    results.reserve(cases.size());
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto result = runWorkload(cases[index], *options, index);
        if (!result.has_value()) {
            return 1;
        }
        results.push_back(*result);
    }
    if (!writeResults(options->outputPath, results)) {
        std::cerr << "could not write benchmark results: " << options->outputPath << '\n';
        return 1;
    }

    std::cout << "Benchmark results written to " << options->outputPath << '\n';
    return 0;
}
