#include "forgecc/cli/command_line.hpp"
#include "forgecc/cli/source_file.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace {

using forgecc::cli::parseCommandLine;

void testValidCompileCommand() {
    const auto result = parseCommandLine({"forgecc", "compile", "input.fcc"});

    assert(result.succeeded());
    assert(result.options->inputPath == std::filesystem::path("input.fcc"));
}

void testInvalidCommandLineArguments() {
    assert(!parseCommandLine({"forgecc"}).succeeded());
    assert(!parseCommandLine({"forgecc", "check", "input.fcc"}).succeeded());
    assert(!parseCommandLine({"forgecc", "compile"}).succeeded());
    assert(!parseCommandLine({"forgecc", "compile", "input.fcc", "extra"}).succeeded());
    assert(!parseCommandLine({"forgecc", "compile", ""}).succeeded());
}

void testSourceFileReading() {
    const auto path = std::filesystem::temp_directory_path() / "forgecc_cli_test.fcc";
    {
        std::ofstream output(path, std::ios::binary);
        output << "A: f32[2, 2]\n\nout = relu(A)\n";
    }

    const auto result = forgecc::cli::readSourceFile(path);
    assert(result.succeeded());
    assert(result.source->contents == "A: f32[2, 2]\n\nout = relu(A)\n");

    std::filesystem::remove(path);
}

void testEmptyAndMissingSourceFiles() {
    const auto emptyPath = std::filesystem::temp_directory_path() / "forgecc_empty.fcc";
    {
        std::ofstream output(emptyPath, std::ios::binary);
    }

    const auto emptyResult = forgecc::cli::readSourceFile(emptyPath);
    assert(emptyResult.succeeded());
    assert(emptyResult.source->contents.empty());
    std::filesystem::remove(emptyPath);

    const auto missing = forgecc::cli::readSourceFile(
        std::filesystem::temp_directory_path() / "forgecc_missing.fcc");
    assert(!missing.succeeded());
    assert(missing.error.find("could not open source file") != std::string::npos);
}

} // namespace

void runCommandLineTests() {
    testValidCompileCommand();
    testInvalidCommandLineArguments();
    testSourceFileReading();
    testEmptyAndMissingSourceFiles();
}
