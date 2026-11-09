#pragma once

#include <filesystem>
#include <string>

namespace forgecc::native {

struct NativeExecutionResult {
    bool success = false;
    int exitCode = -1;
    std::string output;
};

class NativeExecutor {
public:
    [[nodiscard]] NativeExecutionResult execute(
        const std::filesystem::path& executablePath) const;
};

} // namespace forgecc::native
