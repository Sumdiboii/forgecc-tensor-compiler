#include "forgecc/common/diagnostic.hpp"

#include <utility>

namespace forgecc::common {

void DiagnosticEngine::report(DiagnosticSeverity severity, SourceLocation location,
                              std::string message) {
    diagnostics_.push_back(Diagnostic{severity, location, std::move(message)});
}

void DiagnosticEngine::reportError(SourceLocation location, std::string message) {
    report(DiagnosticSeverity::Error, location, std::move(message));
}

bool DiagnosticEngine::hasErrors() const {
    for (const Diagnostic& diagnostic : diagnostics_) {
        if (diagnostic.severity == DiagnosticSeverity::Error) {
            return true;
        }
    }

    return false;
}

const std::vector<Diagnostic>& DiagnosticEngine::diagnostics() const {
    return diagnostics_;
}

} // namespace forgecc::common
