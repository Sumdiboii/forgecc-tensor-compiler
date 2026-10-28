#include "forgecc/common/source_location.hpp"

#include <cassert>

void runSourceLocationTests() {
    const forgecc::common::SourceLocation first;
    const forgecc::common::SourceLocation second{2, 4};

    assert(first.line == 1);
    assert(first.column == 1);
    assert(first != second);
    assert(second == forgecc::common::SourceLocation(2, 4));
}
