#pragma once

#include "forgecc/common/source_location.hpp"

#include <string>
#include <vector>

namespace forgecc::common {

enum class DiagnosticSeverity {
    Error,
    Warning,
};

struct Diagnostic {
    DiagnosticSeverity severity;
    SourceLocation location;
    std::string message;
};

class DiagnosticEngine {
public:
    void report(DiagnosticSeverity severity, SourceLocation location, std::string message);
    void reportError(SourceLocation location, std::string message);

    [[nodiscard]] bool hasErrors() const;
    [[nodiscard]] const std::vector<Diagnostic>& diagnostics() const;

private:
    std::vector<Diagnostic> diagnostics_;
};

} // namespace forgecc::common
