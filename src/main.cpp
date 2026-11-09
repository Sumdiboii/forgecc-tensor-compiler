#include "forgecc/cli/command_line.hpp"
#include "forgecc/cli/source_file.hpp"
#include "forgecc/driver/compiler.hpp"

#include <iostream>
#include <filesystem>
#include <string_view>
#include <unistd.h>
#include <vector>

int main(int argc, char* argv[]) {
    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }

    const auto commandLine = forgecc::cli::parseCommandLine(arguments);
    if (!commandLine.succeeded()) {
        std::cerr << "error: " << commandLine.error << '\n';
        forgecc::cli::printUsage(std::cerr);
        return 2;
    }

    const auto source = forgecc::cli::readSourceFile(commandLine.options->inputPath);
    if (!source.succeeded()) {
        std::cerr << "error: " << source.error << '\n';
        return 1;
    }

    forgecc::driver::CompilerDriver compiler(*source.source);
    std::filesystem::path outputPath;
    if (commandLine.options->outputPath.has_value()) {
        outputPath = *commandLine.options->outputPath;
    } else {
        outputPath = std::filesystem::current_path() /
                     ("forgecc-" + compiler.source().path.stem().string() + "-" +
                      std::to_string(static_cast<long long>(getpid())));
    }

    const auto result = compiler.compileAndRun(outputPath);
    if (!result.success) {
        std::cerr << "error: " << result.diagnostics;
        return 1;
    }

    std::cout << result.output;
    std::cout << "Generated and executed: " << result.executablePath.string() << '\n';
    return 0;
}
