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


// Add this new TEST block to tests/test_ops.cpp

TEST(OpsTest, MatrixMultiplication) {
    // A = | 1  2 |
    //     | 3  4 |
    mtx::Matrix<double> A(2, 2);
    A.at(0, 0) = 1.0; A.at(0, 1) = 2.0;
    A.at(1, 0) = 3.0; A.at(1, 1) = 4.0;

    // B = | 5  6 |
    //     | 7  8 |
    mtx::Matrix<double> B(2, 2);
    B.at(0, 0) = 5.0; B.at(0, 1) = 6.0;
    B.at(1, 0) = 7.0; B.at(1, 1) = 8.0;

    // --- Execute ---
    mtx::Matrix<double> C = A * B;

    // --- Assert ---
    // A * B should be:
    // C = | (1*5 + 2*7)  (1*6 + 2*8) | = | 19  22 |
    //     | (3*5 + 4*7)  (3*6 + 4*8) |   | 43  50 |
    
    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);

    ASSERT_DOUBLE_EQ(C.at(0, 0), 19.0);
    ASSERT_DOUBLE_EQ(C.at(0, 1), 22.0);
    ASSERT_DOUBLE_EQ(C.at(1, 0), 43.0);
    ASSERT_DOUBLE_EQ(C.at(1, 1), 50.0);
}

TEST(OpsTest, ChainedSubtraction) {
    mtx::Matrix<double> A(5, 5);
    mtx::Matrix<double> B(5, 5);
    mtx::Matrix<double> C(5, 5);
    
    // Fill with known values
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            A.at(r, c) = 10.0;
            B.at(r, c) = 2.0;
            C.at(r, c) = 3.0;
        }
    }

    // --- Execute ---
    // Test (A - B) - C
    mtx::Matrix<double> D = A - B - C;

    // --- Assert ---
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            // 10.0 - 2.0 - 3.0 = 5.0
            ASSERT_DOUBLE_EQ(D.at(r, c), 5.0);
        }
    }
}

TEST(OpsTest, Transpose) {
    // A = | 1  2  3 |
    //     | 4  5  6 |
    mtx::Matrix<double> A(2, 3);
    A.at(0, 0) = 1.0; A.at(0, 1) = 2.0; A.at(0, 2) = 3.0;
    A.at(1, 0) = 4.0; A.at(1, 1) = 5.0; A.at(1, 2) = 6.0;

    // --- Test 1: Simple Transpose ---
    // This calls the "evaluating" constructor
    mtx::Matrix<double> B = A.transpose();
    
    ASSERT_EQ(B.rows(), 3); // Original cols
    ASSERT_EQ(B.cols(), 2); // Original rows

    // Check A.at(r, c) == B.at(c, r)
    ASSERT_DOUBLE_EQ(B.at(0, 1), A.at(1, 0)); // 4.0
    ASSERT_DOUBLE_EQ(B.at(2, 0), A.at(0, 2)); // 3.0
    ASSERT_DOUBLE_EQ(B.at(1, 1), A.at(1, 1)); // 5.0
}

TEST(OpsTest, ChainedTranspose) {
    // A (2x3)
    mtx::Matrix<double> A(2, 3);
    A.fill_random(1.0, 5.0);
    
    // B (3x2)
    mtx::Matrix<double> B(3, 2);
    B.fill_random(10.0, 20.0);

    // --- Test 2: Chained Expression ---
    // C = A.transpose() + B
    // This creates a MatrixSum<MatrixTranspose<...>, Matrix<...>>
    // No temporary matrix for A.transpose() should be created!
    mtx::Matrix<double> C = A.transpose() + B;

    ASSERT_EQ(C.rows(), 3);
    ASSERT_EQ(C.cols(), 2);

    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 2; ++c) {
            // Check that C[r][c] = A[c][r] + B[r][c]
            ASSERT_DOUBLE_EQ(C.at(r, c), A.at(c, r) + B.at(r, c));
        }
    }
}

TEST(OpsTest, ScalarOperations) {
    mtx::Matrix<double> A(5, 5);
    A.fill_random(1.0, 5.0);
    
    mtx::Matrix<double> B(5, 5);
    B.fill_random(10.0, 20.0);

    const double scalar = 3.0;

    // --- Test 1: Simple scalar multiply ---
    mtx::Matrix<double> C = A * scalar;
    ASSERT_DOUBLE_EQ(C.at(0, 0), A.at(0, 0) * scalar);
    ASSERT_DOUBLE_EQ(C.at(1, 2), A.at(1, 2) * scalar);

    // --- Test 2: Scalar multiply first ---
    mtx::Matrix<double> D = scalar * A;
    ASSERT_DOUBLE_EQ(D.at(0, 0), scalar * A.at(0, 0));
    ASSERT_DOUBLE_EQ(D.at(2, 1), scalar * A.at(2, 1));

    // --- Test 3: Fully Chained Expression ---
    // E = (A * 3.0) + (3.0 * B)
    // This should all be one lazy expression, evaluated in one pass.
    mtx::Matrix<double> E = (A * scalar) + (scalar * B);
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            ASSERT_DOUBLE_EQ(E.at(r, c), (A.at(r, c) * scalar) + (scalar * B.at(r, c)));
        }
    }
}

TEST(OpsTest, ScalarDivision) {
    mtx::Matrix<double> A(5, 5);
    A.fill_random(100.0, 200.0);
    
    mtx::Matrix<double> B(5, 5);
    B.fill_random(10.0, 20.0);

    const double scalar = 2.0;

    // --- Test 1: Simple scalar division ---
    mtx::Matrix<double> C = A / scalar;
    ASSERT_DOUBLE_EQ(C.at(0, 0), A.at(0, 0) / scalar);

    // --- Test 2: Fully Chained Expression ---
    // E = (A / 2.0) - (B * 3.0)
    mtx::Matrix<double> E = (A / scalar) - (B * 3.0);
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            ASSERT_DOUBLE_EQ(E.at(r, c), (A.at(r, c) / scalar) - (B.at(r, c) * 3.0));
        }
    }
}

TEST(OpsTest, ScalarAddition) {
    mtx::Matrix<double> A(5, 5);
    A.fill_random(10.0, 20.0);

    const double scalar = 100.0;

    // --- Test 1: A + scalar ---
    mtx::Matrix<double> C = A + scalar;
    ASSERT_DOUBLE_EQ(C.at(0, 0), A.at(0, 0) + scalar);

    // --- Test 2: scalar + A ---
    mtx::Matrix<double> D = scalar + A;
    ASSERT_DOUBLE_EQ(D.at(1, 1), scalar + A.at(1, 1));
    
    // --- Test 3: Chained ---
    // E = (A + 100.0) / 2.0
    mtx::Matrix<double> E = (A + scalar) / 2.0;
    ASSERT_DOUBLE_EQ(E.at(2, 2), (A.at(2, 2) + scalar) / 2.0);
}


