#include "forgecc/cli/command_line.hpp"
#include "forgecc/cli/source_file.hpp"
#include "forgecc/driver/compiler.hpp"

#include <iostream>
#include <string_view>
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
    std::cout << "Loaded source file: " << compiler.source().path.string() << '\n';
    std::cout << "Source loading complete; compilation is not implemented yet.\n";
    return 0;
}
