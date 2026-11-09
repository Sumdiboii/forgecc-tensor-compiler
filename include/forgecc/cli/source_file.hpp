#pragma once

#include "forgecc/driver/compiler.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace forgecc::cli {

struct SourceFileReadResult {
    std::optional<driver::SourceInput> source;
    std::string error;

    [[nodiscard]] bool succeeded() const {
        return source.has_value();
    }
};

[[nodiscard]] SourceFileReadResult readSourceFile(const std::filesystem::path& path);

} // namespace forgecc::cli
