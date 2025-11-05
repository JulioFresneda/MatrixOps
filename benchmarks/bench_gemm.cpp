#include <benchmark/benchmark.h>
#include "matrix/Matrix.hpp"

// @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// @@@@@@@@@@@@@@@@@@@@@@@@@@ BENCHMARK TIME @@@@@@@@@@@@@@@@@@@@@@@@@@@
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 

static void BM_MatrixMultiply_Naive(benchmark::State& state) {
    int size = state.range(0); 
    
    // Setup: Create two matrices
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    // This is the loop that Google Benchmark runs
    for (auto _ : state) {
        // --- This is the code being measured ---
        mtx::Matrix<double> C = A * B;
        // ----------------------------------------

        // Tell the compiler not to optimize away our result,
        // which it might do if C is unused.
        benchmark::DoNotOptimize(C);
    }
}

static void BM_MatrixMultiply_Reordered(benchmark::State& state) {
    int size = state.range(0);
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    for (auto _ : state) {
        // --- Call our NEW function ---
        mtx::Matrix<double> C = mtx::gemm_v1_reordered(A, B);
        // ---------------------------
        benchmark::DoNotOptimize(C);
    }
}

static void BM_MatrixMultiply_Tiled(benchmark::State& state) {
    int size = state.range(0);
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    for (auto _ : state) {
        // --- Call our NEW Tiled function ---
        mtx::Matrix<double> C = mtx::gemm_v2_tiling(A, B);
        // ---------------------------------
        benchmark::DoNotOptimize(C);
    }
}

static void BM_MatrixMultiply_Tiled_Raw(benchmark::State& state) {
    int size = state.range(0);
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    for (auto _ : state) {
        // --- Call our NEW Tiled + Raw function ---
        mtx::Matrix<double> C = mtx::gemm_v3_tiled_raw(A, B);
        // ---------------------------------------
        benchmark::DoNotOptimize(C);
    }
}

static void BM_MatrixMultiply_Tiled_Raw_SIMD(benchmark::State& state) {
    int size = state.range(0);
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    for (auto _ : state) {
        // --- Call our NEW Tiled + Raw + SIMD function ---
        mtx::Matrix<double> C = mtx::gemm_v4_tiled_raw_simd(A, B);
        // ---------------------------------------------
        benchmark::DoNotOptimize(C);
    }
}


static void BM_MatrixMultiply_Multithreaded(benchmark::State& state) {
    int size = state.range(0);
    mtx::Matrix<double> A(size, size);
    mtx::Matrix<double> B(size, size);
    A.fill_random(0.0, 1.0);
    B.fill_random(0.0, 1.0);

    for (auto _ : state) {
        mtx::Matrix<double> C = mtx::gemm_v5_multithreaded(A, B);
        // ---------------------------------------
        benchmark::DoNotOptimize(C);
    }
}



BENCHMARK(BM_MatrixMultiply_Naive)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMultiply_Reordered)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMultiply_Tiled)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMultiply_Tiled_Raw)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMultiply_Tiled_Raw_SIMD)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatrixMultiply_Multithreaded)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond)
    // GBenchmark must know
    ->UseRealTime();

// This line starts the benchmark program
BENCHMARK_MAIN();