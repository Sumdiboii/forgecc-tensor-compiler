#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <ostream>

namespace forgecc::cli {

struct CommandLineOptions {
    std::filesystem::path inputPath;
    std::optional<std::filesystem::path> outputPath;
};

struct CommandLineParseResult {
    std::optional<CommandLineOptions> options;
    std::string error;

    [[nodiscard]] bool succeeded() const {
        return options.has_value();
    }
};

[[nodiscard]] CommandLineParseResult parseCommandLine(
    const std::vector<std::string_view>& arguments);
void printUsage(std::ostream& output);

} // namespace forgecc::cli
