add_executable(forgecc_tests
    unit/common/diagnostic_test.cpp
    unit/common/source_location_test.cpp
    unit/cli/command_line_test.cpp
    unit/codegen/cpp_emitter_test.cpp
    unit/codegen/standalone_emitter_test.cpp
    unit/frontend/lexer_test.cpp
    unit/frontend/parser_test.cpp
    unit/frontend/part2_parser_test.cpp
    unit/frontend/program_ast_test.cpp
    unit/frontend/declaration_parser_test.cpp
    unit/frontend/program_parser_test.cpp
    unit/semantic/type_and_symbol_table_test.cpp
    unit/semantic/semantic_analyzer_test.cpp
    unit/ir/graph_test.cpp
    unit/lowering/lowerer_test.cpp
    unit/lowering/loop_ir_test.cpp
    unit/lowering/loop_lowerer_test.cpp
    unit/memory/memory_plan_test.cpp
    unit/native/native_compiler_test.cpp
    unit/optimizer/pass_manager_test.cpp
    unit/optimizer/dead_code_elimination_test.cpp
    unit/optimizer/common_subexpression_elimination_test.cpp
    unit/test_main.cpp
)
target_link_libraries(forgecc_tests PRIVATE forgecc forgecc_cli_support)
add_test(NAME forgecc_tests COMMAND forgecc_tests)
add_test(NAME forgecc_cli_compile
         COMMAND forgecc_cli compile ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/basic.fcc)
add_test(NAME forgecc_cli_missing_source COMMAND forgecc_cli compile)
set_tests_properties(forgecc_cli_missing_source PROPERTIES WILL_FAIL TRUE)
