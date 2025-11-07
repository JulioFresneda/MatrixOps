cmake_minimum_required(VERSION 3.15)

project(MatrixOps VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(matrix_ops INTERFACE)
target_include_directories(matrix_ops INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(matrix_ops INTERFACE -mavx2)

find_package(OpenMP REQUIRED)
target_link_libraries(matrix_ops INTERFACE OpenMP::OpenMP_CXX)

add_executable(run_app src/examples.cpp)

target_link_libraries(run_app PRIVATE matrix_ops)

# GoogleTest setup
enable_testing()
find_package(GTest REQUIRED)
add_executable(run_tests tests/test_io.cpp tests/test_ops.cpp tests/test_linalg.cpp)
target_link_libraries(run_tests PRIVATE matrix_ops GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(run_tests)

find_package(benchmark REQUIRED)
add_executable(run_benchmarks benchmarks/bench_gemm.cpp)
target_link_libraries(run_benchmarks PRIVATE matrix_ops benchmark::benchmark)