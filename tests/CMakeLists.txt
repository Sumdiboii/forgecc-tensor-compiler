add_executable(forgecc_tests
    unit/common/diagnostic_test.cpp
    unit/common/source_location_test.cpp
    unit/frontend/lexer_test.cpp
    unit/frontend/parser_test.cpp
    unit/frontend/part2_parser_test.cpp
    unit/frontend/program_ast_test.cpp
    unit/frontend/declaration_parser_test.cpp
    unit/test_main.cpp
)
target_link_libraries(forgecc_tests PRIVATE forgecc)
add_test(NAME forgecc_tests COMMAND forgecc_tests)
