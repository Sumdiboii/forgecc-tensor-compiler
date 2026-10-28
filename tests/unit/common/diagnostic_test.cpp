#include "forgecc/common/diagnostic.hpp"

#include <cassert>

void runDiagnosticTests() {
    forgecc::common::DiagnosticEngine diagnostics;
    const forgecc::common::SourceLocation location{3, 7};

    diagnostics.reportError(location, "unexpected character");

    assert(diagnostics.hasErrors());
    assert(diagnostics.diagnostics().size() == 1);
    assert(diagnostics.diagnostics().front().location == location);
    assert(diagnostics.diagnostics().front().message == "unexpected character");
}
