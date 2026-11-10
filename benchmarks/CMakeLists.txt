add_executable(forgecc_benchmarks cpu_benchmarks.cpp)
target_link_libraries(forgecc_benchmarks PRIVATE forgecc)

add_executable(forgecc_report report_performance.cpp)

if(FORGECC_BUILD_TESTS)
    add_test(NAME forgecc_report_valid
             COMMAND forgecc_report
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/test_data/minimal.csv
                     --output ${CMAKE_CURRENT_BINARY_DIR}/minimal-report.md)
    add_test(NAME forgecc_report_malformed
             COMMAND forgecc_report
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/test_data/malformed.csv
                     --output ${CMAKE_CURRENT_BINARY_DIR}/malformed-report.md)
    set_tests_properties(forgecc_report_malformed PROPERTIES WILL_FAIL TRUE)
endif()
