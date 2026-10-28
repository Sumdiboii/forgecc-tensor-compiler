cmake_minimum_required(VERSION 3.20)

project(ForgeCC VERSION 0.1.0 LANGUAGES CXX)

add_library(forgecc STATIC
    src/common/diagnostic.cpp
    src/frontend/lexer.cpp
)
target_compile_features(forgecc PUBLIC cxx_std_20)
target_include_directories(forgecc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

option(FORGECC_BUILD_TESTS "Build ForgeCC tests" OFF)
option(FORGECC_BUILD_BENCHMARKS "Build ForgeCC benchmarks" OFF)

if(FORGECC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FORGECC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
