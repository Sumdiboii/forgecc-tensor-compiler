#include "forgecc/cli/source_file.hpp"

#include <fstream>
#include <iterator>

namespace forgecc::cli {

SourceFileReadResult readSourceFile(const std::filesystem::path& path) {
    if (path.empty()) {
        return {std::nullopt, "source-file path must not be empty"};
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {std::nullopt, "could not open source file: " + path.string()};
    }

    std::string contents(std::istreambuf_iterator<char>(input), {});
    if (input.bad()) {
        return {std::nullopt, "could not read source file: " + path.string()};
    }

    return {driver::SourceInput{path, std::move(contents)}, {}};
}

} // namespace forgecc::cli
