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
    return std::find(kImplementations.begin(), kImplementations.end(), implementation) !=
           kImplementations.end();
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
    output << "## Implementation comparison\n\n"
              "Speedup is baseline median divided by compared median; values above 1.0 are faster.\n\n"
              "| Workload | Dimensions | Baseline median (ns) | Graph IR median (ns) | Loop IR median (ns) | Graph speedup | Loop speedup |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const double baseline = statistics(workload.implementations.at("baseline"))->median;
        const double optimized = statistics(workload.implementations.at("optimized"))->median;
        const double forgecc = statistics(workload.implementations.at("forgecc"))->median;
        output << "| " << name << " | " << workload.dimensions << " | " << std::fixed
               << std::setprecision(2) << baseline << " | " << optimized << " | " << forgecc
               << " | " << baseline / optimized << " | " << baseline / forgecc << " |\n";
    }
    output << "\n| Workload | Graph IR difference vs baseline | Loop IR difference vs baseline |\n"
              "|---|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const double baseline = statistics(workload.implementations.at("baseline"))->median;
        const double optimized = statistics(workload.implementations.at("optimized"))->median;
        const double forgecc = statistics(workload.implementations.at("forgecc"))->median;
        output << "| " << name << " | " << performanceDifference(baseline, optimized) << " | "
               << performanceDifference(baseline, forgecc) << " |\n";
    }
    output << "\n";
}

void writeScalingTable(std::ostream& output, const ReportData& report) {
    output << "## Scaling observations\n\n"
              "The work-unit column is an observed normalization only; it is not a complexity proof.\n\n"
              "| Workload | Implementation | Dimensions | Work units | Median (ns) | ns/work unit |\n"
              "|---|---|---|---:|---:|---:|\n";
    for (const auto& [name, workload] : report.workloads) {
        const std::size_t work = workUnits(workload);
        for (const std::string_view implementation : kImplementations) {
            const double median = statistics(workload.implementations.at(std::string(implementation)))
                                      ->median;
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
              "This report was generated from saved Phase 9.2 CSV measurements. Each workload\n"
              "uses the same deterministic inputs for the straightforward C++ baseline, the\n"
              "Graph IR implementation after the existing DCE/CSE passes, and the Loop IR\n"
              "generated implementation. Warm-up calls are excluded from timing; median kernel\n"
              "execution time is the primary comparison metric.\n\n"
           << "Input CSV: `" << input.string() << "`\n\n";
    writeComparisonTable(output, report);
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
              "## Conclusions\n\n"
              "The tables above report measured differences only. Explanations of scaling or\n"
              "speedups should be treated as hypotheses until validated with additional runs\n"
              "and, where available, hardware-counter measurements.\n";
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
