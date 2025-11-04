#include <gtest/gtest.h>
#include "matrix/Matrix.hpp" // Your main library header
#include <iostream>

// Test simple (A + B)
TEST(OpsTest, SimpleAddition) {
    size_t rows = 10;
    size_t cols = 10;

    mtx::Matrix<double> A(rows, cols);
    mtx::Matrix<double> B(rows, cols);

    // Fill with known values
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            A.at(r, c) = 1.0;
            B.at(r, c) = 2.0;
        }
    }

    // --- Execute ---
    // This calls the evaluating constructor
    // with a MatrixSum<...> expression
    mtx::Matrix<double> C = A + B;

    // --- Assert ---
    ASSERT_EQ(C.rows(), rows);
    ASSERT_EQ(C.cols(), cols);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            // A(1.0) + B(2.0) should be 3.0
            ASSERT_DOUBLE_EQ(C.at(r, c), 3.0);
        }
    }
}

// Test chained (A + B + C)
TEST(OpsTest, ChainedAddition) {
    size_t rows = 5;
    size_t cols = 5;

    mtx::Matrix<double> A(rows, cols);
    mtx::Matrix<double> B(rows, cols);
    mtx::Matrix<double> C(rows, cols);
    
    // Fill with known values
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            A.at(r, c) = 1.0;
            B.at(r, c) = 2.0;
            C.at(r, c) = 3.0;
        }
    }

    // --- Execute ---
    // This is the real test!
    // It builds an expression tree: MatrixSum<MatrixSum<A, B>, C>
    // It should all be evaluated in ONE pass.
    std::cout << "Testing chained addition (A + B + C)..." << std::endl;
    mtx::Matrix<double> D = A + B + C;

    // --- Assert ---
    ASSERT_EQ(D.rows(), rows);
    ASSERT_EQ(D.cols(), cols);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            // A(1.0) + B(2.0) + C(3.0) should be 6.0
            ASSERT_DOUBLE_EQ(D.at(r, c), 6.0);
            
            // A more robust check:
            ASSERT_DOUBLE_EQ(D.at(r, c), A.at(r, c) + B.at(r, c) + C.at(r, c));
        }
    }
}