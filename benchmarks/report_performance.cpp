#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::array<std::string_view, 3> kImplementations = {
    "baseline", "optimized", "forgecc"};

struct Options {
    std::filesystem::path inputPath;
    std::filesystem::path outputPath;
};

struct ImplementationData {
    std::size_t warmupIterations = 0;
    std::size_t measurementIterations = 0;
    std::map<std::size_t, double> samples;
    double compilationMilliseconds = 0.0;
    std::size_t sourceBytes = 0;
    std::uintmax_t executableBytes = 0;
};

struct WorkloadData {
    std::string name;
    std::string dimensions;
    std::map<std::string, ImplementationData> implementations;
};

struct ReportData {
    std::map<std::string, WorkloadData> workloads;
};

struct Statistics {
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

std::optional<std::size_t> parseSize(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t consumed = 0;
        const std::size_t value = std::stoull(std::string(text), &consumed);
        if (consumed != text.size()) {
            return std::nullopt;
        }
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<double> parseNumber(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t consumed = 0;
        const double value = std::stod(std::string(text), &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            return std::nullopt;
        }
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<std::string> parseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field += '"';
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field += character;
        }
    }
    if (quoted) {
        return {};
    }
    fields.push_back(std::move(field));
    return fields;
}

std::optional<Statistics> statistics(const ImplementationData& implementation) {
    if (implementation.samples.empty()) {
        return std::nullopt;
    }
    std::vector<double> samples;
    samples.reserve(implementation.samples.size());
    for (const auto& [index, sample] : implementation.samples) {
        static_cast<void>(index);
        samples.push_back(sample);
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
                              ? (samples[middle - 1] + samples[middle]) / 2.0
                              : samples[middle];
    return Statistics{median, samples.front(), samples.back()};
}

bool isImplementation(std::string_view implementation) {
    if (std::find(kImplementations.begin(), kImplementations.end(), implementation) !=
        kImplementations.end()) {
        return true;
    }
    constexpr std::string_view prefix = "tiled-";
    if (!implementation.starts_with(prefix)) {
        return false;
    }
    const auto tileSize = parseSize(implementation.substr(prefix.size()));
    return tileSize.has_value() && *tileSize != 0;
}

std::vector<std::string> tiledImplementations(const ReportData& report) {
    std::map<std::string, bool> names;
    for (const auto& [workloadName, workload] : report.workloads) {
        static_cast<void>(workloadName);
        for (const auto& [implementation, data] : workload.implementations) {
            static_cast<void>(data);
            if (implementation.starts_with("tiled-")) {
                names.emplace(implementation, true);
            }
        }
    }
    std::vector<std::string> result;
    result.reserve(names.size());
    for (const auto& [name, present] : names) {
        static_cast<void>(present);
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> reportImplementations(const ReportData& report) {
    std::vector<std::string> result;
    for (const std::string_view implementation : kImplementations) {
        result.emplace_back(implementation);
    }
    const auto tiled = tiledImplementations(report);
    result.insert(result.end(), tiled.begin(), tiled.end());
    return result;
}

std::optional<std::vector<std::size_t>> parseDimensions(std::string_view text);

std::optional<ReportData> readResults(const std::filesystem::path& path, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open benchmark CSV: " + path.string();
        return std::nullopt;
    }

    std::string line;
    if (!std::getline(input, line)) {
        error = "benchmark CSV is empty";
        return std::nullopt;
    }
    const std::vector<std::string> header = parseCsvLine(line);
    const std::vector<std::string> required = {
        "workload",       "implementation", "dimensions",          "warmup_iterations",
        "measurement_iterations", "sample_index", "sample_ns",      "average_ns",
        "median_ns",      "min_ns",          "max_ns",              "compile_ms",
        "source_bytes",   "executable_bytes", "speedup_vs_baseline"};
    std::map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) {
        columns.emplace(header[index], index);
    }
    for (const std::string& name : required) {
        if (!columns.contains(name)) {
            error = "benchmark CSV is missing column: " + name;
            return std::nullopt;
        }
    }

    ReportData report;
    std::size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = parseCsvLine(line);
        if (fields.size() != header.size()) {
            error = "invalid CSV field count on line " + std::to_string(lineNumber);
            return std::nullopt;
        }
        const auto field = [&fields, &columns](std::string_view name) -> const std::string& {
            return fields.at(columns.at(std::string(name)));
        };

        const std::string workloadName = field("workload");
        const std::string implementation = field("implementation");
        const std::string dimensions = field("dimensions");
        if (workloadName.empty() || dimensions.empty() || !isImplementation(implementation) ||
            !parseDimensions(dimensions)) {
            error = "invalid workload or implementation on line " + std::to_string(lineNumber);
            return std::nullopt;
        }
        const auto warmup = parseSize(field("warmup_iterations"));
        const auto measurements = parseSize(field("measurement_iterations"));
        const auto sampleIndex = parseSize(field("sample_index"));
        const auto sample = parseNumber(field("sample_ns"));
        const auto average = parseNumber(field("average_ns"));
        const auto median = parseNumber(field("median_ns"));
        const auto minimum = parseNumber(field("min_ns"));
        const auto maximum = parseNumber(field("max_ns"));
        const auto compileMilliseconds = parseNumber(field("compile_ms"));
        const auto sourceBytes = parseSize(field("source_bytes"));
        const auto executableBytes = parseSize(field("executable_bytes"));
        const auto speedup = parseNumber(field("speedup_vs_baseline"));
        if (!warmup || !measurements || *measurements == 0 || !sampleIndex || !sample ||
            !average || !median || !minimum || !maximum || !compileMilliseconds ||
            !sourceBytes || !executableBytes || !speedup || *sample < 0.0 ||
            *compileMilliseconds < 0.0 || *average < 0.0 || *median < 0.0 || *minimum < 0.0 ||
            *maximum < 0.0 || *sampleIndex >= *measurements) {
            error = "invalid numeric value on line " + std::to_string(lineNumber);
            return std::nullopt;
        }

        WorkloadData& workload = report.workloads[workloadName];
        if (workload.name.empty()) {
            workload.name = workloadName;
            workload.dimensions = dimensions;
        } else if (workload.dimensions != dimensions) {
            error = "inconsistent dimensions for workload: " + workloadName;
            return std::nullopt;
        }
        ImplementationData& data = workload.implementations[implementation];
        if (!data.samples.empty() &&
            (data.warmupIterations != *warmup || data.measurementIterations != *measurements ||
             data.compilationMilliseconds != *compileMilliseconds ||
             data.sourceBytes != *sourceBytes || data.executableBytes != *executableBytes)) {
            error = "inconsistent metadata for " + workloadName + "/" + implementation;
            return std::nullopt;
        }
        data.warmupIterations = *warmup;
        data.measurementIterations = *measurements;
        data.compilationMilliseconds = *compileMilliseconds;
        data.sourceBytes = *sourceBytes;
        data.executableBytes = *executableBytes;
        if (!data.samples.emplace(*sampleIndex, *sample).second) {
            error = "duplicate sample index for " + workloadName + "/" + implementation;
            return std::nullopt;
        }
    }

    if (report.workloads.empty()) {
        error = "benchmark CSV contains no measurements";
        return std::nullopt;
    }
    for (const auto& [name, workload] : report.workloads) {
        for (const std::string_view implementation : kImplementations) {
            const auto found = workload.implementations.find(std::string(implementation));
            if (found == workload.implementations.end() ||
                found->second.samples.size() != found->second.measurementIterations) {
                error = "missing measurements for " + name + "/" + std::string(implementation);
                return std::nullopt;
            }
            for (std::size_t index = 0; index < found->second.measurementIterations; ++index) {
                if (!found->second.samples.contains(index)) {
                    error = "missing sample index for " + name + "/" +
                            std::string(implementation);
                    return std::nullopt;
                }
            }
        }
    }
    return report;
}

std::optional<std::vector<std::size_t>> parseDimensions(std::string_view text) {
    std::vector<std::size_t> result;
    bool inBracket = false;
    std::string number;
    for (const char character : text) {
        if (character == '[') {
            if (inBracket) {
                return std::nullopt;
            }
            inBracket = true;
        } else if (character == ']') {
            if (!inBracket || number.empty()) {
                return std::nullopt;
            }
            const auto value = parseSize(number);
            if (!value || *value == 0) {
                return std::nullopt;
            }
            result.push_back(*value);
            number.clear();
            inBracket = false;
        } else if (inBracket && character == 'x') {
            if (!number.empty()) {
                const auto value = parseSize(number);
                if (!value || *value == 0) {
                    return std::nullopt;
                }
                result.push_back(*value);
                number.clear();
            } else {
                return std::nullopt;
            }
        } else if (inBracket && character >= '0' && character <= '9') {
            number += character;
        } else {
            return std::nullopt;
        }
    }
    if (inBracket || !number.empty() || result.empty()) {
        return std::nullopt;
    }
    return result;
}

std::size_t workUnits(const WorkloadData& workload) {
    const auto parsedDimensions = parseDimensions(workload.dimensions);
    if (!parsedDimensions) {
        return 0;
    }
    const std::vector<std::size_t>& values = *parsedDimensions;
    std::size_t result = 1;
    for (const std::size_t value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value) {
            return 0;
        }
        result *= value;
    }
    if (workload.dimensions.find("][") != std::string::npos && values.size() == 4) {
        return values[0] * values[1] * values[3];
    }
    return result;
}

std::string performanceDifference(double baseline, double compared) {
    const double percentage = (compared - baseline) / baseline * 100.0;
    std::ostringstream result;
    result << std::fixed << std::setprecision(2) << percentage << "%";
    return result.str();
}

void writeComparisonTable(std::ostream& output, const ReportData& report) {
    const auto implementations = reportImplementations(report);
    output << "## Implementation comparison\n\n"
              "Speedup is baseline median divided by compared median; values above 1.0 are faster.\n\n"
              "| Workload | Dimensions |";
    for (const std::string& implementation : implementations) {
        output << " " << implementation << " median (ns) |";
    }
    output << "\n|---|---:|";
    for (const std::string& implementation : implementations) {
        static_cast<void>(implementation);
        output << "---:|";
    }
    output << "\n";
    for (const auto& [name, workload] : report.workloads) {
        output << "| " << name << " | " << workload.dimensions << " |";
        for (const std::string& implementation : implementations) {
            const auto found = workload.implementations.find(implementation);
            if (found == workload.implementations.end()) {
                output << " unavailable |";
            } else {
                output << " " << std::fixed << std::setprecision(2)
                       << statistics(found->second)->median << " |";
            }
        }
        output << "\n";
    }
    output << "\n| Workload |";
    for (const std::string& implementation : implementations) {
        if (implementation != "baseline") {
            output << " " << implementation << " difference vs baseline |";
        }
    }
    output << "\n|---|";
    for (const std::string& implementation : implementations) {
        if (implementation != "baseline") {
            output << "---:|";
        }
    }
    output << "\n";
    for (const auto& [name, workload] : report.workloads) {
        const double baseline = statistics(workload.implementations.at("baseline"))->median;
        output << "| " << name << " |";
        for (const std::string& implementation : implementations) {
            if (implementation == "baseline") {
                continue;
            }
            const auto found = workload.implementations.find(implementation);
            if (found == workload.implementations.end()) {
                output << " unavailable |";
            } else {
                output << " " << performanceDifference(
                                      baseline, statistics(found->second)->median)
                       << " |";
            }
        }
        output << "\n";
    }
    output << "\n";
}

std::optional<double> speedup(double reference, double candidate) {
    if (candidate <= 0.0) {
        return std::nullopt;
    }
    return reference / candidate;
}

std::string formatSpeedup(std::optional<double> value) {
    if (!value.has_value()) {
        return "n/a";
    }
    std::ostringstream result;
    result << std::fixed << std::setprecision(3) << *value;
    return result.str();
}

void writeOptimizationAnalysis(std::ostream& output, const ReportData& report) {
    const auto implementations = reportImplementations(report);
    output << "## Relative speedups\n\n"
              "Speedup is reference median divided by candidate median; values above 1.0 are faster.\n\n"
              "| Workload | Implementation | Median (ns) | Vs baseline | Vs Loop IR |\n"
              "|---|---|---:|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const double baseline = statistics(workload.implementations.at("baseline"))->median;
        const double loop = statistics(workload.implementations.at("forgecc"))->median;
        for (const std::string& implementation : implementations) {
            if (implementation == "baseline") {
                continue;
            }
            const auto found = workload.implementations.find(implementation);
            if (found == workload.implementations.end()) {
                continue;
            }
            const double median = statistics(found->second)->median;
            output << "| " << name << " | " << implementation << " | " << std::fixed
                   << std::setprecision(2) << median << " | "
                   << formatSpeedup(speedup(baseline, median)) << " | "
                   << formatSpeedup(speedup(loop, median)) << " |\n";
        }
    }
    output << "\n## Best implementation summary\n\n"
              "Speedup is reference median divided by candidate median; values above 1.0 are faster.\n"
              "The `forgecc` column is the baseline Loop IR implementation; tiled columns apply to matmul workloads.\n\n"
              "| Workload | Baseline median (ns) | Loop IR median (ns) | Best implementation | Best tiled variant | Best vs baseline | Best vs Loop IR |\n"
              "|---|---:|---:|---|---|---:|---:|\n";

    std::size_t tiledWorkloads = 0;
    std::size_t tiledWinsOverLoop = 0;
    for (const auto& [name, workload] : report.workloads) {
        const auto tiledNames = [&workload] {
            std::vector<std::string> names;
            for (const auto& [implementation, data] : workload.implementations) {
                static_cast<void>(data);
                if (implementation.starts_with("tiled-")) {
                    names.push_back(implementation);
                }
            }
            return names;
        }();
        if (!tiledNames.empty()) {
            ++tiledWorkloads;
        }

        const double baseline = statistics(workload.implementations.at("baseline"))->median;
        const double loop = statistics(workload.implementations.at("forgecc"))->median;
        std::string bestImplementation = "baseline";
        double bestMedian = baseline;
        for (const std::string& implementation : implementations) {
            const auto found = workload.implementations.find(implementation);
            if (found == workload.implementations.end()) {
                continue;
            }
            const double median = statistics(found->second)->median;
            if (median < bestMedian) {
                bestMedian = median;
                bestImplementation = implementation;
            }
        }

        std::string bestTiled = "n/a";
        double bestTiledMedian = 0.0;
        for (const std::string& implementation : tiledNames) {
            const double median = statistics(workload.implementations.at(implementation))->median;
            if (bestTiled == "n/a" || median < bestTiledMedian) {
                bestTiled = implementation;
                bestTiledMedian = median;
            }
        }
        if (!tiledNames.empty() && bestTiledMedian < loop) {
            ++tiledWinsOverLoop;
        }

        output << "| " << name << " | " << std::fixed << std::setprecision(2) << baseline
               << " | " << loop << " | " << bestImplementation << " | " << bestTiled
               << " | ";
        const auto bestVsBaseline = speedup(baseline, bestMedian);
        const auto bestVsLoop = speedup(loop, bestMedian);
        output << formatSpeedup(bestVsBaseline) << " | " << formatSpeedup(bestVsLoop) << " |\n";
    }
    output << "\n";
    if (tiledWorkloads == 0) {
        output << "No tiled matmul workloads were present in the input CSV.\n\n";
        return;
    }
    output << "Across " << tiledWorkloads << " matmul workloads, the best tested tiled variant "
           << (tiledWinsOverLoop == tiledWorkloads ? "was faster than" : "was not faster than")
           << " the baseline Loop IR for " << tiledWinsOverLoop
           << " workload(s). This is a measured observation for this run, not a universal "
              "tile-size recommendation.\n\n";
}

void writeScalingTable(std::ostream& output, const ReportData& report) {
    output << "## Scaling observations\n\n"
              "The work-unit column is an observed normalization only; it is not a complexity proof.\n\n"
              "| Workload | Implementation | Dimensions | Work units | Median (ns) | ns/work unit |\n"
              "|---|---|---|---:|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const std::size_t work = workUnits(workload);
        for (const std::string& implementation : reportImplementations(report)) {
            const auto found = workload.implementations.find(implementation);
            if (found == workload.implementations.end()) {
                continue;
            }
            const double median = statistics(found->second)->median;
            output << "| " << name << " | " << implementation << " | " << workload.dimensions
                   << " | " << work << " | " << std::fixed << std::setprecision(2) << median
                   << " | " << (work == 0 ? 0.0 : median / static_cast<double>(work)) << " |\n";
        }
    }
    output << "\nMeasured observations should be compared across rows with the same implementation and\n"
              "workload family; machine noise, allocation behavior, and compiler output can affect\n"
              "individual measurements.\n\n";
}

void writeArtifactTable(std::ostream& output, const ReportData& report) {
    output << "## Compilation and artifacts\n\n"
              "These values are separate from steady-state execution timing.\n\n"
              "| Workload | Compilation (ms) | Generated source (bytes) | Executable (bytes) |\n"
              "|---|---:|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const ImplementationData& data = workload.implementations.at("baseline");
        output << "| " << name << " | " << std::fixed << std::setprecision(3)
               << data.compilationMilliseconds << " | " << data.sourceBytes << " | "
               << data.executableBytes << " |\n";
    }
    output << "\n";
}

bool writeReport(const std::filesystem::path& path, const std::filesystem::path& input,
                 const ReportData& report) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "# ForgeCC Performance Report\n\n"
              "## Methodology\n\n"
              "This report was generated from saved Phase 10.3 CSV measurements. Each workload\n"
              "uses the same deterministic inputs for the straightforward C++ baseline, the\n"
              "Graph IR implementation after the existing DCE/CSE passes, and the Loop IR\n"
              "generated implementation, including tiled Loop IR variants for matmul. Warm-up\n"
              "calls are excluded from timing; median kernel execution time is the primary\n"
              "comparison metric. Compilation time is measured separately.\n\n"
           << "Input CSV: `" << input.string() << "`\n\n";
    if (!report.workloads.empty()) {
        const auto& firstWorkload = report.workloads.begin()->second;
        const auto baseline = firstWorkload.implementations.find("baseline");
        if (baseline != firstWorkload.implementations.end()) {
            output << "Configured warmup iterations: " << baseline->second.warmupIterations
                   << "; measured iterations: " << baseline->second.measurementIterations
                   << ". Raw per-iteration samples are retained in the CSV.\n\n";
        }
    }
    writeComparisonTable(output, report);
    writeOptimizationAnalysis(output, report);
    writeScalingTable(output, report);
    writeArtifactTable(output, report);
    output << "## Limitations\n\n"
              "- Compilation time is measured separately and is not included in kernel timings.\n"
              "- The benchmark consumes output through a checksum; this report does not replace\n"
              "  the element-by-element correctness tests.\n"
              "- Results are CPU- and system-dependent, and small workloads are sensitive to\n"
              "  scheduling and timer resolution.\n"
              "- Hardware counters are not collected automatically. If Linux `perf` is available,\n"
              "  it can be used separately, for example: `perf stat -e cycles,instructions,cache-misses\n"
              "  ./build/benchmarks/forgecc_benchmarks ...`. Counter availability depends on\n"
              "  kernel permissions and the host CPU.\n\n"
              "## Cache and locality interpretation\n\n"
              "The baseline accesses `B[k, j]` with a row stride as the reduction index changes.\n"
              "The tiled implementation reuses smaller row, column, and shared-dimension blocks,\n"
              "and its innermost column loop accesses rows of `B` and `C` contiguously. This is\n"
              "expected to improve spatial and temporal locality when a tile fits useful cache\n"
              "levels, but this report contains timing measurements only. It does not measure L1,\n"
              "L2, or L3 cache events.\n\n"
              "## Conclusions\n\n"
              "The tables above report measured differences only. Small workloads may remain faster\n"
              "with the baseline because tiled-loop and boundary overhead dominates. Larger\n"
              "workloads may benefit from cache blocking, but the best tile size is workload- and\n"
              "machine-dependent. Additional repeated runs and hardware counters are needed to\n"
              "separate locality effects from system noise.\n";
    return output.good();
}

std::optional<Options> parseOptions(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input" || argument == "--output") {
            if (index + 1 >= argc) {
                return std::nullopt;
            }
            if (argument == "--input") {
                options.inputPath = argv[++index];
            } else {
                options.outputPath = argv[++index];
            }
        } else if (argument == "--help") {
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    }
    if (options.inputPath.empty() || options.outputPath.empty()) {
        return std::nullopt;
    }
    return options;
}

} // namespace

int main(int argc, char* argv[]) {
    const auto options = parseOptions(argc, argv);
    if (!options) {
        std::cerr << "Usage: forgecc_report --input RESULTS.csv --output REPORT.md\n";
        return 2;
    }

    std::string error;
    const auto report = readResults(options->inputPath, error);
    if (!report) {
        std::cerr << error << '\n';
        return 1;
    }
    std::error_code filesystemError;
    if (!options->outputPath.parent_path().empty()) {
        std::filesystem::create_directories(options->outputPath.parent_path(), filesystemError);
        if (filesystemError) {
            std::cerr << "could not create report directory: " << filesystemError.message() << '\n';
            return 1;
        }
    }
    if (!writeReport(options->outputPath, options->inputPath, *report)) {
        std::cerr << "could not write report: " << options->outputPath << '\n';
        return 1;
    }
    std::cout << "Performance report written to " << options->outputPath << '\n';
    return 0;
}
