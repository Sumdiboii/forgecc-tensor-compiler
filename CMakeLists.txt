cmake_minimum_required(VERSION 3.20)

project(ForgeCC VERSION 0.1.0 LANGUAGES CXX)

add_library(forgecc STATIC
    src/common/diagnostic.cpp
    src/codegen/cpp_emitter.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/semantic_analyzer.cpp
    src/ir/graph.cpp
    src/lowering/lowerer.cpp
    src/lowering/loop_ir.cpp
    src/lowering/loop_lowerer.cpp
    src/memory/liveness.cpp
    src/memory/memory_plan.cpp
    src/optimizer/pass_manager.cpp
    src/optimizer/dead_code_elimination.cpp
    src/optimizer/common_subexpression_elimination.cpp
    src/semantic/symbol_table.cpp
)
target_compile_features(forgecc PUBLIC cxx_std_20)
target_include_directories(forgecc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(forgecc_cli_support STATIC
    src/cli/command_line.cpp
    src/cli/source_file.cpp
)
target_include_directories(forgecc_cli_support PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(forgecc_cli src/main.cpp)
set_target_properties(forgecc_cli PROPERTIES OUTPUT_NAME forgecc)
target_link_libraries(forgecc_cli PRIVATE forgecc forgecc_cli_support)

option(FORGECC_BUILD_TESTS "Build ForgeCC tests" OFF)
option(FORGECC_BUILD_BENCHMARKS "Build ForgeCC benchmarks" OFF)

if(FORGECC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FORGECC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
