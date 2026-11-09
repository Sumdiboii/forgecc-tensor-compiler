#pragma once

#include <filesystem>
#include <string>
#include <utility>

namespace forgecc::driver {

struct SourceInput {
    std::filesystem::path path;
    std::string contents;
};

class CompilerDriver {
public:
    explicit CompilerDriver(SourceInput source) : source_(std::move(source)) {}

    [[nodiscard]] const SourceInput& source() const {
        return source_;
    }

private:
    SourceInput source_;
};

} // namespace forgecc::driver
