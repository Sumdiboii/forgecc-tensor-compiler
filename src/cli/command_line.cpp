#include "forgecc/cli/command_line.hpp"

namespace forgecc::cli {

CommandLineParseResult parseCommandLine(const std::vector<std::string_view>& arguments) {
    if (arguments.size() <= 1) {
        return {std::nullopt, "missing command"};
    }
    if (arguments[1] != "compile") {
        return {std::nullopt, "unsupported command: " + std::string(arguments[1])};
    }
    if (arguments.size() <= 2) {
        return {std::nullopt, "missing source-file path"};
    }
    if (arguments.size() > 3) {
        return {std::nullopt, "unexpected additional arguments"};
    }
    if (arguments[2].empty()) {
        return {std::nullopt, "source-file path must not be empty"};
    }

    return {CommandLineOptions{std::filesystem::path(arguments[2])}, {}};
}

void printUsage(std::ostream& output) {
    output << "Usage:\n  forgecc compile <input.fcc>\n";
}

} // namespace forgecc::cli
