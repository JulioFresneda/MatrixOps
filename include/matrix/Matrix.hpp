#pragma once

#include <vector>
#include <iostream>
#include <fstream>   // For std::ofstream
#include <random>    // For std::mt19937
#include <stdexcept> // For std::runtime_error
#include <iomanip>   // For std::setprecision
#include <type_traits> // For std:is_floating_point_v
#include <immintrin.h> // For AVX2 (SIMD) Intrinsics
#include <omp.h> // for OpenMP
#include <algorithm> // for std::swap
#include <numeric>    // For std::iota
#include <cmath>      // For std::abs

#include "MatrixSum.hpp"
#include "MatrixSubtract.hpp"
#include "MatrixTranspose.hpp"
#include "MatrixScalarMultiply.hpp"
#include "MatrixScalarDivide.hpp"
#include "MatrixScalarAdd.hpp"

namespace mtx {

template<typename T>
class Matrix : public MatrixExpression<Matrix<T>, T>{
public:

    // Returns a direct, "unsafe" pointer to the raw data
    // for high-performance operations.
    T* raw_data() { return data.data(); }
    const T* raw_data() const { return data.data(); }

    // --> Base Constructor <--
    Matrix(size_t rows, size_t cols) {
        num_rows = rows;
        num_cols = cols;
        data.resize(rows * cols);
        
        if (rows == 0 || cols == 0) {
            throw std::runtime_error("Matrix dimensions must be non-zero.");
        }
    }

    // --> Binary Constructor <-- Fastest way to load large matrices
    Matrix(const std::string& binary_path) {
        std::ifstream file(binary_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + binary_path);
        }

        // 1. Read the dimensions
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

        // 2. Resize. It allocates memory but does not initialize values yet. It contains noise for now
        data.resize(num_rows * num_cols);

        // 3. Read the data block directly into the vector's buffer
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));

        // ########################## Why reinterpret_cast? ##########################
        // ---------------------------------------------------------------------------
        // The binary write method only accepts a pointer to char (byte array).
        // Since we are trying to write numbers, we need to tell the compiler:
        // "Hey! This info are char type (they are not)! 
        // Write it even if this is nonsense in human eyes"
        // And that is why this binary is not human-readable :)
        // ---------------------------------------------------------------------------
    }

    // --> CSV Constructor <-- 
    Matrix(const std::string& csv_path, char delimiter) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + csv_path);
        }

        std::vector<std::vector<T>> temp_data;
        std::string line;
        
        size_t cols = 0;

        // --- Pass 1: Read data into temporary 2D vector ---
        while (std::getline(file, line)) {
            if (line.empty()) continue; // Skip empty lines

            std::vector<T> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, delimiter)) {
                // Convert string "cell" to type T (float, double, etc.)
                row.push_back(static_cast<T>(std::stod(cell))); 
            }

            if (temp_data.empty()) {
                // This is the first row, set our column count
                cols = row.size(); 
            } else if (row.size() != cols) {
                // Check for malformed CSV (rows with different lengths)
                throw std::runtime_error("Malformed CSV: Inconsistent row lengths.");
            }
            
            temp_data.push_back(row);
        }

        if (temp_data.empty()) {
            throw std::runtime_error("CSV file is empty.");
        }

        // --- Pass 2: Copy data to our final flat vector ---
        num_rows = temp_data.size();
        num_cols = cols;
        data.resize(num_rows * num_cols);

        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                at(r, c) = temp_data[r][c];
            }
        }
    }

    
    
    // Non-const version (for writing)
    T& at(size_t r, size_t c) {
        if (r >= num_rows || c >= num_cols) {
            throw std::out_of_range("Matrix access out of bounds.");
        }
        return data[r * num_cols + c];
    }

    // Const version (for reading)
    const T& at(size_t r, size_t c) const {
        if (r >= num_rows || c >= num_cols) {
            throw std::out_of_range("Matrix access out of bounds.");
        }
        return data[r * num_cols + c];
    }

    // --- Getters ---
    size_t rows() const { return num_rows; }
    size_t cols() const { return num_cols; }


    // Fills the matrix with random numbers between min and max
    void fill_random(T min, T max) {
        std::random_device rd;  // Hardware-based random seed -> Truly random
        std::mt19937 gen(rd()); // Mersenne Twister engine -> Deterministic and extremely fast
        
        if constexpr (std::is_floating_point_v<T>) {
            // T is float or double, use the REAL distribution
            std::uniform_real_distribution<T> dist(min, max);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = dist(gen);
            }
        } else {
            // T is int, long, etc., use the INTEGER distribution
            std::uniform_int_distribution<T> dist(min, max);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = dist(gen);
            }
        }
    }

    void save_csv(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + path);
        }

        // In a CSV, a precision of 10 is enough for our purposes
        file << std::fixed << std::setprecision(10);

        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                file << at(r, c); // No need for use const bc this is a const method
                if (c < num_cols - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
    }

    // Binary are faster but not readable by humans
    void save_binary(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + path);
        }

        // In binaries we have to write the dimensions first
        file.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
        file.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));

        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));

        // ########################## Why reinterpret_cast? ##########################
        // ---------------------------------------------------------------------------
        // The binary write method only accepts a pointer to char (byte array).
        // Since we are trying to write numbers, we need to tell the compiler:
        // "Hey! This info are char type (they are not)! 
        // Write it even if this is nonsense in human eyes"
        // And that is why this binary is not human-readable :)
        // ---------------------------------------------------------------------------

        

    }

    template<typename E>
    Matrix(const MatrixExpression<E, T>& expr)
        : num_rows(expr.rows()), num_cols(expr.cols()), data(expr.rows() * expr.cols()) 
    {   
        std::cout << "Evaluating expression!" << std::endl;
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                // ########################## AWESOME THING HAPPENING HERE ##########################
                // ----------------------------------------------------------------------------------
                // It calls at(r,c) on the expression,
                // which calculates A(r,c) + B(r,c) + C(r,c) ...
                // and we store the final result. ONE loop!
                // ----------------------------------------------------------------------------------
                at(r, c) = expr.at(r, c);
            }
        }
    }

    
    // ########################################################################
    // ------------------------- Operator * Overloads -------------------------
    // ########################################################################
    // Inside, not like +, bc we won't use this operator with templates
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // -------------------------- NAIVE VERSION - O(N^3) ----------------------
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    Matrix<T> operator*(const Matrix<T>& other) const {
        if constexpr (std::is_same_v<T, double>) {
            return gemm_v5_multithreaded(*this, other);  
        } else {
            return gemm_v3_tiled_raw(*this, other);
        }
    }

    void print(const std::string& title) const {
        std::cout << "--- " << title << " (" << num_rows << "x" << num_cols << ") ---\n";
        std::cout << std::fixed << std::setprecision(3);
   
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                std::cout << std::setw(8) << at(r, c) << " ";
            }
            std::cout << "\n";



        }
        std::cout << "---------------------------------------\n" << std::endl;
    }

    auto transpose() const {
        return mtx::MatrixTranspose<Matrix<T>, T>(*this);
    }

    void swap_rows(size_t r1, size_t r2) {
        if (r1 == r2) return; // No-op

        T* pA = raw_data();
        const size_t n_cols = cols();

        // Use raw pointers for a fast, C-style row swap
        for (size_t c = 0; c < n_cols; ++c) {
            std::swap(pA[r1 * n_cols + c], pA[r2 * n_cols + c]);
        }
    }

    // --- LINALG DETERMINANT ---
    // Calculates the determinant of the matrix.
    // Returns 0.0 if the matrix is singular (not invertible).
    T determinant() const {
        if (num_rows != num_cols) {
            throw std::runtime_error("Determinant requires a square matrix.");
        }
        Matrix<T> A_copy(*this);
        std::vector<size_t> P;

        // 3. Decompose and get swap count
        int swap_count = lup_decomposition_inplace(A_copy, P);

        // 4. Check for singularity
        if (swap_count == -1) {
            return static_cast<T>(0.0);
        }

        // 5. Calculate determinant from diagonal
        T det = static_cast<T>(1.0);
        for (size_t i = 0; i < num_rows; ++i) {
            det *= A_copy.at(i, i);
        }

        // 6. Apply sign flip from swaps
        if (swap_count % 2 != 0) {
            return -det;
        } else {
            return det;
        }
    }

    // --- LINALG RANK ---
    // Calculates the rank of the matrix.
    // The rank is the number of linearly independent rows (or columns).
    size_t rank(T tolerance = 1e-9) const {
        // 1. Create a copy to decompose
        Matrix<T> A_copy(*this);
        std::vector<size_t> P;

        // --- THIS IS THE FIX ---
        // We run the decomposition. We don't care about the return
        // value (whether it's singular or not).
        // The decomposition will run as far as it can.
        lup_decomposition_inplace(A_copy, P);

        // 3. The rank *is* the number of non-zero pivots on the
        //    diagonal of the resulting U matrix.
        size_t rank = 0;
        for (size_t i = 0; i < num_rows; ++i) {
            // We check if the absolute value is greater than a small
            // tolerance to handle floating-point inaccuracies.
            if (std::abs(A_copy.at(i, i)) > tolerance) {
                rank++;
            }
        }
        
        return rank;
    }

    std::vector<T> solve(const std::vector<T>& b) const {
        if (num_rows != num_cols) {
            throw std::runtime_error("Solve requires a square matrix.");
        }
        if (num_rows != b.size()) {
            throw std::runtime_error("Solve: matrix and vector dimensions do not match.");
        }

        // 1. Decompose a copy of this matrix
        Matrix<T> A_copy(*this);
        std::vector<size_t> P;
        int swap_count = lup_decomposition_inplace(A_copy, P);

        // 2. Check for singularity
        if (swap_count == -1) {
            throw std::runtime_error("Cannot solve: Matrix is singular.");
        }

        // 3. Solve using the decomposed matrix
        return lup_solve(A_copy, P, b);
    }

    // --- LINALG INVERSE ---
    // Calculates the inverse of the matrix (A_inv)
    // such that A * A_inv = Identity.
    Matrix<T> inverse() const {
        if (num_rows != num_cols) {
            throw std::runtime_error("Inverse requires a square matrix.");
        }
        
        const size_t n = num_rows;

        // 1. Decompose a copy of this matrix
        Matrix<T> A_copy(*this);
        std::vector<size_t> P;
        int swap_count = lup_decomposition_inplace(A_copy, P);

        // 2. Check for singularity
        if (swap_count == -1) {
            throw std::runtime_error("Cannot invert: Matrix is singular.");
        }

        // 3. Create the output inverse matrix
        Matrix<T> A_inv(n, n); // Starts as all zeros

        // 4. Solve for each column of the inverse
        std::vector<T> b(n, 0.0); // The "b" vector (a column of I)

        for (size_t j = 0; j < n; ++j) {
            // Create the j-th column of the Identity matrix
            b[j] = 1.0;
            if (j > 0) {
                b[j-1] = 0.0;
            }

            // 5. Solve for the j-th column of the inverse
            std::vector<T> x = lup_solve(A_copy, P, b);

            // 6. Place the solution vector 'x' into the j-th column
            //    of the inverse matrix
            for (size_t i = 0; i < n; ++i) {
                A_inv.at(i, j) = x[i];
            }
        }
        
        return A_inv;
    }



private:
    size_t num_rows;
    size_t num_cols;

    // We will use a 1D vector to store the 2D data internally.
    // We use size_t for extremely large matrices.
    std::vector<T> data; 
};

// @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ########################################################################@
// ------------------------- Operator   Overloads -------------------------@
// ########################################################################@
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @

template<typename E1, typename E2, typename T>
auto operator+(const MatrixExpression<E1, T>& lhs, const MatrixExpression<E2, T>& rhs) {
    // This doesn't calculate anything!
    return mtx::MatrixSum<E1, E2, T>(
        *static_cast<const E1*>(&lhs), 
        *static_cast<const E2*>(&rhs)
    );
}

template<typename E1, typename E2, typename T>
auto operator-(const MatrixExpression<E1, T>& lhs, const MatrixExpression<E2, T>& rhs) {
    return mtx::MatrixSubtract<E1, E2, T>(
        *static_cast<const E1*>(&lhs), 
        *static_cast<const E2*>(&rhs)
    );
}

template<typename E, typename T>
auto operator*(const MatrixExpression<E, T>& expr, T scalar) {
    return mtx::MatrixScalarMultiply<E, T>(
        *static_cast<const E*>(&expr), 
        scalar
    );
}

// Handles: scalar * Matrix (e.g., 5.0 * A)
template<typename E, typename T>
auto operator*(T scalar, const MatrixExpression<E, T>& expr) {
    // Multiplication is commutative, so we can just re-use the other operator
    return expr * scalar;
}

template<typename E, typename T>
auto operator/(const MatrixExpression<E, T>& expr, T scalar) {
    return mtx::MatrixScalarDivide<E, T>(
        *static_cast<const E*>(&expr), 
        scalar
    );
}

// Handles: Matrix + scalar (e.g., A + 5.0)
template<typename E, typename T>
auto operator+(const MatrixExpression<E, T>& expr, T scalar) {
    return mtx::MatrixScalarAdd<E, T>(
        *static_cast<const E*>(&expr), 
        scalar
    );
}

// Handles: scalar + Matrix (e.g., 5.0 + A)
template<typename E, typename T>
auto operator+(T scalar, const MatrixExpression<E, T>& expr) {
    // Addition is commutative
    return expr + scalar;
}

// Handles: Matrix - scalar (e.g., A - 5.0)
template<typename E, typename T>
auto operator-(const MatrixExpression<E, T>& expr, T scalar) {
    // We just call operator+ with the negative scalar!
    // This re-uses the MatrixScalarAdd class.
    return expr + (-scalar);
}

// Handles: scalar - Matrix (e.g., 5.0 - A)
template<typename E, typename T>
auto operator-(T scalar, const MatrixExpression<E, T>& expr) {
    // This is (scalar + (A * -1.0))
    // We re-use operator+ and operator*
    return scalar + (expr * static_cast<T>(-1.0));
}

// Still O(n^3), but highly optimized for the CPU cache.
// The naive (r,c,k) loop "column-hops" on Matrix B, causing massive cache misses.
// This (r,k,c) loop accesses B and C in continuous rows, making it cache-friendly.

// Looks complicated but we just changed the order. Given C = A * B:
// Naive version:
//  > C11 = A11 * B11
//  > C11 += A12 * B21 <- Column jump!!! Inefficient
//  > C12 = A11 * B12
//  > C12 += A12 * B22 <- Column jump!!! Inefficient
// Improved naive version:
//  > C11 = A11 * B11
//  > C12 = A11 * B12 <- No column jump :) Efficient
//  > C11 += A12 * B21
//  > C12 += A12 * B22 <- No column jump :) Efficient

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ------------------- IMPROVED NAIVE VERSION - O(N^3) --------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

template<typename T>
Matrix<T> gemm_naive(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());

    // The classic (r, c, k) triple-loop
    for (size_t r = 0; r < C.rows(); ++r) {     
        for (size_t c = 0; c < C.cols(); ++c) { 
            T sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) { 
                sum += A.at(r, k) * B.at(k, c);
            }
            C.at(r, c) = sum;
        }
    }
    return C;
}

template<typename T>
Matrix<T> gemm_v1_reordered(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());

    // --- THE "r, k, c" OPTIMIZED LOOP ---
    for (size_t r = 0; r < A.rows(); ++r) {
        for (size_t k = 0; k < A.cols(); ++k) {
            // Get A[r][k] once, it's constant for the inner loop
            T a_rk = A.at(r, k); 
            for (size_t c = 0; c < B.cols(); ++c) {
                // C[r][c] += A[r][k] * B[k][c]
                // C.at(r, c) -> Accesses memory in a line (GOOD)
                // B.at(k, c) -> Accesses memory in a line (GOOD!)
                C.at(r, c) += a_rk * B.at(k, c);
            }
        }
    }
    return C;
}


// till O(n^3), but massively faster due to cache-tiling.
// This algorithm processes the matrix in small blocks (tiles) that fit in the L1 cache.
// This minimizes "cache misses" (slow RAM access) and keeps the CPU fully loaded.
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- CPU CACHE VERSION - O(N^3) ---------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
template<typename T>
Matrix<T> gemm_v2_tiling(const Matrix<T>& A, const Matrix<T>& B) {

    // -> HOW MUCH MEMORY IN YOUR L1 CACHE? <-
    // Define a tile size. 32x32 is common.
    // 32*32*sizeof(double) = 8KB. 
    // We need 3 tiles (A, B, C) = 24KB, which fits in our 32KB L1 cache.
    const int TILE_SIZE = 32;

    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());

    for (size_t r = 0; r < A.rows(); r += TILE_SIZE) {
        for (size_t k = 0; k < A.cols(); k += TILE_SIZE) {
            for (size_t c = 0; c < B.cols(); c += TILE_SIZE) {
                
                // --- This is the "naive" loop, but on a tiny tile ---
                // We also use "blocking" on the loop bounds
                for (size_t r_tile = r; r_tile < std::min(r + TILE_SIZE, A.rows()); ++r_tile) {
                    for (size_t k_tile = k; k_tile < std::min(k + TILE_SIZE, A.cols()); ++k_tile) {
                        
                        T a_rk = A.at(r_tile, k_tile); // Get A[r][k] once

                        for (size_t c_tile = c; c_tile < std::min(c + TILE_SIZE, B.cols()); ++c_tile) {
                            C.at(r_tile, c_tile) += a_rk * B.at(k_tile, c_tile);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// Like previous version, but we eliminate all function call overhead.
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- CPU CACHE VERSION - O(N^3) ---------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
template<typename T>
Matrix<T> gemm_v3_tiled_raw(const Matrix<T>& A, const Matrix<T>& B) {
    const int TILE_SIZE = 32;

    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());
    
    // Get dimensions and raw data pointers ONCE
    const size_t A_rows = A.rows();
    const size_t A_cols = A.cols();
    const size_t B_cols = B.cols();

    const T* pA = A.raw_data();
    const T* pB = B.raw_data();
          T* pC = C.raw_data();

    

    for (size_t r = 0; r < A_rows; r += TILE_SIZE) {
        for (size_t k = 0; k < A_cols; k += TILE_SIZE) {
            for (size_t c = 0; c < B_cols; c += TILE_SIZE) {
                
                // --- Tiled loop with raw pointer math ---
                for (size_t r_tile = r; r_tile < std::min(r + TILE_SIZE, A_rows); ++r_tile) {
                    for (size_t k_tile = k; k_tile < std::min(k + TILE_SIZE, A_cols); ++k_tile) {
                        
                        // Get A[r][k]
                        // No function call, no bounds check
                        const T a_rk = pA[r_tile * A_cols + k_tile]; 

                        for (size_t c_tile = c; c_tile < std::min(c + TILE_SIZE, B_cols); ++c_tile) {
                            // C[r][c] += A[r][k] * B[k][c]
                            // No function calls, just raw pointer math. This is FAST.
                            pC[r_tile * B_cols + c_tile] += a_rk * pB[k_tile * B_cols + c_tile];
                        }
                    }
                }
            }
        }
    }
    return C;
}


// --------------------------- WHAT IS SIMD? -----------------------------------
// SIMD stands for "Single Instruction, Multiple Data". It's a CPU feature
// that allows it to perform one operation (like "multiply") on a whole
// "vector" of data (e.g., 4 doubles) at the same time, in one clock cycle.
//
// --------------------------- WHAT IS AVX? -------------------------------------
// AVX (Advanced Vector Extensions) is the name of the SIMD instruction
// set in modern CPUs. We are using AVX2, which has 256-bit registers.
// A 'double' is 64 bits, so 256 / 64 = 4. We can operate on 4 doubles at once.
//
// --------------------------- WHY IS THIS TYPE-SPECIFIC? -----------------------
// These CPU instructions are the "bare metal". We have to use the
// *exact* instruction for the data type.
// - `__m256d` and `_mm256_..._pd` are for "packed doubles".
// - `__m256`  and `_mm256_..._ps` are for "packed singles" (floats).
// - Integer intrinsics are completely different and more complex.
// This function is hard-coded to speak the "double" language to the CPU.
//
// --------------------------- WHY NOT INTEGERS? --------------------------------
// NOT Possible. Integer SIMD is very different.
// Matrix multiplication involves multiply-and-add, which can easily overflow an int. 
// High-performance integer multiplication uses different, much more complex instructions 
// and is a whole separate challenge. 
// We'll stick to floats and doubles, which is what SIMD is primarily designed for in math.
// Yeah I know, double mults easier than int mults, wtf xD
//
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- CPU SIMD VERSION - O(N^3) ---------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CAUTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- ONLY DOUBLES SUPPORTED -------------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

template<typename T>
Matrix<T> gemm_v4_tiled_raw_simd(const Matrix<T>& A, const Matrix<T>& B) {
    // This optimization only works for doubles.
    // We use a static_assert to stop compilation if T is not double.
    static_assert(std::is_same_v<T, double>, "SIMD version currently only supports double.");

    // Use a smaller tile size for SIMD, 16 is often good
    const int TILE_SIZE = 16; 

    // AVX registers hold 4 doubles (256 bits / 64 bits = 4)
    const int SIMD_STEP = 4;

    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());
    
    const size_t A_rows = A.rows();
    const size_t A_cols = A.cols();
    const size_t B_cols = B.cols();

    const double* pA = A.raw_data();
    const double* pB = B.raw_data();
            double* pC = C.raw_data();

    

    for (size_t r = 0; r < A_rows; r += TILE_SIZE) {
        for (size_t k = 0; k < A_cols; k += TILE_SIZE) {
            for (size_t c = 0; c < B_cols; c += TILE_SIZE) {
                
                for (size_t r_tile = r; r_tile < std::min(r + TILE_SIZE, A_rows); ++r_tile) {
                    for (size_t k_tile = k; k_tile < std::min(k + TILE_SIZE, A_cols); ++k_tile) {
                        
                        // Load A[r][k] into all 4 "lanes" of a SIMD register
                        // a_vec = [a_rk, a_rk, a_rk, a_rk]
                        __m256d a_vec = _mm256_set1_pd(pA[r_tile * A_cols + k_tile]);

                        // --- Vectorized Inner Loop ---
                        // We step 4 columns at a time
                        for (size_t c_tile = c; c_tile < std::min(c + TILE_SIZE, B_cols); c_tile += SIMD_STEP) {
                            
                            // Check if we have 4 elements left.
                            // This is a simple (but slow) way to handle edges.
                            if (c_tile + SIMD_STEP > std::min(c + TILE_SIZE, B_cols)) {
                                // Not enough elements for a full SIMD op,
                                // run the old "scalar" loop for the remainder
                                for(size_t c_scalar = c_tile; c_scalar < std::min(c + TILE_SIZE, B_cols); ++c_scalar) {
                                    pC[r_tile * B_cols + c_scalar] += pA[r_tile * A_cols + k_tile] * pB[k_tile * B_cols + c_scalar];
                                }
                                break; // Exit SIMD loop
                            }

                            // 1. Load 4 doubles from C
                            // c_vec = [C[r][c], C[r][c+1], C[r][c+2], C[r][c+3]]
                            __m256d c_vec = _mm256_loadu_pd(&pC[r_tile * B_cols + c_tile]);
                            
                            // 2. Load 4 doubles from B
                            // b_vec = [B[k][c], B[k][c+1], B[k][c+2], B[k][c+3]]
                            __m256d b_vec = _mm256_loadu_pd(&pB[k_tile * B_cols + c_tile]);

                            // 3. Multiply and Add (Fused Multiply-Add - FMA)
                            // c_vec += a_vec * b_vec
                            // This does 4 muls and 4 adds in one instruction!
                            c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));

                            // 4. Store 4 results back into C
                            _mm256_storeu_pd(&pC[r_tile * B_cols + c_tile], c_vec);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ EFFICIENCY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- MULTITHREAD VERSION - O(N^3) -------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CAUTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ----------------------- ONLY DOUBLES SUPPORTED -------------------------
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

template<typename T>
Matrix<T> gemm_v5_multithreaded(const Matrix<T>& A, const Matrix<T>& B) {
    static_assert(std::is_same_v<T, double>, "SIMD version currently only supports double.");

    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix multiplication dimension mismatch.");
    }

    Matrix<T> C(A.rows(), B.cols());
    
    const size_t A_rows = A.rows();
    const size_t A_cols = A.cols();
    const size_t B_cols = B.cols();

    const double* pA = A.raw_data();
    const double* pB = B.raw_data();
            double* pC = C.raw_data();

    const int TILE_SIZE = 16;
    const int SIMD_STEP = 4;

    // ---> PARALLELISM HERE <---
    #pragma omp parallel for
    for (size_t r = 0; r < A_rows; r += TILE_SIZE) {
        for (size_t k = 0; k < A_cols; k += TILE_SIZE) {
            for (size_t c = 0; c < B_cols; c += TILE_SIZE) {
                
                for (size_t r_tile = r; r_tile < std::min(r + TILE_SIZE, A_rows); ++r_tile) {
                    for (size_t k_tile = k; k_tile < std::min(k + TILE_SIZE, A_cols); ++k_tile) {
                        
                        __m256d a_vec = _mm256_set1_pd(pA[r_tile * A_cols + k_tile]);

                        for (size_t c_tile = c; c_tile < std::min(c + TILE_SIZE, B_cols); c_tile += SIMD_STEP) {
                            
                            if (c_tile + SIMD_STEP > std::min(c + TILE_SIZE, B_cols)) {
                                for(size_t c_scalar = c_tile; c_scalar < std::min(c + TILE_SIZE, B_cols); ++c_scalar) {
                                    pC[r_tile * B_cols + c_scalar] += pA[r_tile * A_cols + k_tile] * pB[k_tile * B_cols + c_scalar];
                                }
                                break; 
                            }

                            __m256d c_vec = _mm256_loadu_pd(&pC[r_tile * B_cols + c_tile]);
                            __m256d b_vec = _mm256_loadu_pd(&pB[k_tile * B_cols + c_tile]);
                            c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));
                            _mm256_storeu_pd(&pC[r_tile * B_cols + c_tile], c_vec);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ---
// ## LINALG KERNEL 1: LU DECOMPOSITION (Doolittle's Algorithm)
// ---
//
// WHAT IS THIS?
// This is an O(n^3) algorithm that factors a square matrix 'A'
// into two triangular matrices: 'L' (Lower) and 'U' (Upper).
// A = L * U
//
// WHY?
// This is the "workhorse" for solving linear algebra problems.
// Once you have L and U, finding the determinant, inverse,
// or solving a system of equations (Ax=b) becomes very fast.
//
// IMPLEMENTATION:
// We perform an "in-place" decomposition. The function takes
// a matrix and overwrites its data. The resulting matrix
// will store *both* L and U in a single, combined matrix.
// - The Upper triangle (including diagonal) is 'U'.
// - The Lower triangle (excluding diagonal) is 'L'.
// - The diagonal of 'L' is implicitly all 1s.

// This function modifies 'A' in-place and returns 'true' on success.
// NOTE: This simple version does NOT use pivoting, so it will
// fail (divide by zero) for many matrices.
template<typename T>
bool lu_decomposition_inplace(Matrix<T>& A) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("LU decomposition requires a square matrix.");
    }

    const size_t n = A.rows();
    T* pA = A.raw_data(); // Get raw pointer for max speed

    for (size_t i = 0; i < n; ++i) { // i = current row/column "pivot"
        
        // --- 1. Calculate 'U' for the current row 'i' ---
        // U[i][k] = A[i][k] - sum(L[i][j] * U[j][k]) for j < i
        for (size_t k = i; k < n; ++k) { // k = column
            T sum = 0;
            for (size_t j = 0; j < i; ++j) { // j = inner sum
                // L[i][j] is pA[i*n + j]
                // U[j][k] is pA[j*n + k]
                sum += pA[i * n + j] * pA[j * n + k];
            }
            // U[i][k] = A[i][k] - sum
            pA[i * n + k] = pA[i * n + k] - sum;
        }

        // --- 2. Calculate 'L' for the current column 'i' ---
        // L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i])) / U[i][i] for j < i
        for (size_t k = i + 1; k < n; ++k) { // k = row
            T sum = 0;
            for (size_t j = 0; j < i; ++j) { // j = inner sum
                // L[k][j] is pA[k*n + j]
                // U[j][i] is pA[j*n + i]
                sum += pA[k * n + j] * pA[j * n + i];
            }
            
            // Check for divide by zero (singularity)
            // U[i][i] is pA[i*n + i]
            if (pA[i * n + i] == 0) {
                // This matrix cannot be decomposed without pivoting
                // (or it's singular)
                return false; 
            }

            // L[k][i] = (A[k][i] - sum) / U[i][i]
            pA[k * n + i] = (pA[k * n + i] - sum) / pA[i * n + i];
        }
    }
    
    return true; // Success
}

// ---
// ## LINALG KERNEL 1: LUP DECOMPOSITION (w/ Partial Pivoting)
// ---
//
// This is the robust O(n^3) Crout algorithm. It finds the
// L, U, and P matrices such that P*A = L*U.
//
// - A: Is modified in-place to store L and U.
// - P: Is an output vector that records the row swaps.
//
// Returns 'true' on success and 'false' if the matrix is singular.

// ---
// ## LINALG KERNEL 1: LUP DECOMPOSITION (w/ Partial Pivoting)
// ---
//
// This is the robust O(n^3) Crout algorithm. It finds the
// L, U, and P matrices such that P*A = L*U.
//
// - A: Is modified in-place to store L (lower triangle, excl. diagonal)
//      and U (upper triangle, incl. diagonal).
// - P: Is an output vector that records the row swaps.
//
// Returns:
// - (int) swap_count: The number of row swaps performed (success).
// - -1: If the matrix is singular (failure).

template<typename T>
int lup_decomposition_inplace(Matrix<T>& A, std::vector<size_t>& P) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("LUP decomposition requires a square matrix.");
    }

    const size_t n = A.rows();
    int swap_count = 0; // Tracks row swaps for the determinant sign
    
    // Initialize permutation vector P to [0, 1, 2, ..., n-1]
    P.resize(n);
    std::iota(P.begin(), P.end(), 0); // Fills P with 0, 1, 2...

    for (size_t k = 0; k < n; ++k) { // For each column/pivot
        
        // --- 1. Find Pivot ---
        // Find the row with the largest value in this column (k)
        // starting from the current row (k) down.
        size_t max_row = k;
        T max_val = std::abs(A.at(k, k));

        for (size_t i = k + 1; i < n; ++i) {
            T val = std::abs(A.at(i, k));
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // --- 2. Check for Singularity ---
        // If the max value in the column is 0, the matrix is singular.
        // (Tolerance check '1e-9' is good for floats/doubles)
        if (max_val < 1e-9) { 
            return -1; // Failure: Matrix is singular
        }

        // --- 3. Perform Pivoting (Row Swap) ---
        if (max_row != k) {
            // Record the swap in our permutation vector
            std::swap(P[k], P[max_row]);
            // And swap the rows in the actual matrix
            A.swap_rows(k, max_row);
            swap_count++; // Increment swap counter
        }

        // --- 4. Calculate L and U for this column ---
        
        // Calculate L column (below the pivot)
        // L[i][k] = U[i][k] / U[k][k]
        for (size_t i = k + 1; i < n; ++i) {
            A.at(i, k) = A.at(i, k) / A.at(k, k);
        }
        
        // Update the remaining submatrix
        // A[i][j] = A[i][j] - L[i][k] * U[k][j]
        for (size_t i = k + 1; i < n; ++i) {
            for (size_t j = k + 1; j < n; ++j) {
                A.at(i, j) = A.at(i, j) - A.at(i, k) * A.at(k, j);
            }
        }
    }
    
    return swap_count; // Success
}

// ---
// ## LINALG KERNEL 2: LUP SOLVER
// ---
//
// Solves the system Ax = b using a pre-computed LUP decomposition.
// - A_lu: The in-place matrix containing L and U.
// - P: The permutation vector from LUP.
// - b: The "right-hand side" vector.
//
// Returns: The solution vector 'x'.

template<typename T>
std::vector<T> lup_solve(const Matrix<T>& A_lu, const std::vector<size_t>& P, const std::vector<T>& b) {
    const size_t n = A_lu.rows();
    if (n != b.size()) {
        throw std::runtime_error("LUP solve: matrix and vector dimensions do not match.");
    }

    std::vector<T> x(n);
    std::vector<T> y(n);

    // --- 1. Forward Substitution (Solves Ly = Pb) ---
    // This solves for the intermediate vector 'y'
    for (size_t i = 0; i < n; ++i) {
        T sum = 0;
        for (size_t j = 0; j < i; ++j) {
            // L[i][j] is A_lu.at(i, j)
            sum += A_lu.at(i, j) * y[j];
        }
        // y[i] = b[P[i]] - sum
        // (L's diagonal is 1s, so no division needed)
        y[i] = b[P[i]] - sum;
    }

    // --- 2. Backward Substitution (Solves Ux = y) ---
    // This solves for the final vector 'x'
    // We must loop from the last row (n-1) up to 0
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (size_t j = i + 1; j < n; ++j) {
            // U[i][j] is A_lu.at(i, j)
            sum += A_lu.at(i, j) * x[j];
        }
        // x[i] = (y[i] - sum) / U[i][i]
        x[i] = (y[i] - sum) / A_lu.at(i, i);
    }

    return x;
}


} // namespace mtx