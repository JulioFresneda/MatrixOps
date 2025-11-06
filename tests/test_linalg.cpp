#include <gtest/gtest.h>
#include "matrix/Matrix.hpp" // Your main library header
#include <iostream>

TEST(LinalgTest, Determinant) {
    // A = | 1  2 |
    //     | 3  4 |
    // det(A) = (1*4) - (2*3) = 4 - 6 = -2
    mtx::Matrix<double> A(2, 2);
    A.at(0, 0) = 1.0; A.at(0, 1) = 2.0;
    A.at(1, 0) = 3.0; A.at(1, 1) = 4.0;
    
    ASSERT_DOUBLE_EQ(A.determinant(), -2.0);

    // B = | 3  8  4 |
    //     | 2  1  1 |
    //     | 5  2  7 |
    // det(B) = 3(7-2) - 8(14-5) + 4(4-5)
    //        = 3(5) - 8(9) + 4(-1)
    //        = 15 - 72 - 4 = -61
    mtx::Matrix<double> B(3, 3);
    B.at(0, 0) = 3; B.at(0, 1) = 8; B.at(0, 2) = 4;
    B.at(1, 0) = 2; B.at(1, 1) = 1; B.at(1, 2) = 1;
    B.at(2, 0) = 5; B.at(2, 1) = 2; B.at(2, 2) = 7;

    ASSERT_DOUBLE_EQ(B.determinant(), -61.0);

    // Test the pivot case
    // C = | 0  1 |
    //     | 1  1 |
    // det(C) = (0*1) - (1*1) = -1
    mtx::Matrix<double> C(2, 2);
    C.at(0, 0) = 0.0; C.at(0, 1) = 1.0;
    C.at(1, 0) = 1.0; C.at(1, 1) = 1.0;
    
    ASSERT_DOUBLE_EQ(C.determinant(), -1.0);

    // Test a singular matrix
    // D = | 1  1 |
    //     | 1  1 |
    // det(D) = 0
    mtx::Matrix<double> D(2, 2);
    D.at(0, 0) = 1.0; D.at(0, 1) = 1.0;
    D.at(1, 0) = 1.0; D.at(1, 1) = 1.0;
    
    ASSERT_DOUBLE_EQ(D.determinant(), 0.0);
}

TEST(LinalgTest, Rank) {
    // A = | 1  2 |
    //     | 3  4 |
    // det is -2 (non-zero), so rank is 2
    mtx::Matrix<double> A(2, 2);
    A.at(0, 0) = 1.0; A.at(0, 1) = 2.0;
    A.at(1, 0) = 3.0; A.at(1, 1) = 4.0;
    
    ASSERT_EQ(A.rank(), 2);

    // B = | 1  1  1 |
    //     | 2  2  2 |
    //     | 3  3  3 |
    // Row 2 = 2*Row 1, Row 3 = 3*Row 1. Only 1 independent row.
    // Rank is 1.
    mtx::Matrix<double> B(3, 3);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            B.at(r, c) = r + 1;
        }
    }
    ASSERT_EQ(B.rank(), 1);
    
    // C = | 1  1 |
    //     | 1  1 |
    // Singular, but not all zeros. Rank is 1.
    mtx::Matrix<double> C(2, 2);
    C.at(0, 0) = 1.0; C.at(0, 1) = 1.0;
    C.at(1, 0) = 1.0; C.at(1, 1) = 1.0;
    
    ASSERT_EQ(C.rank(), 1);
}

TEST(LinalgTest, Solve) {
    // System:
    // 2x + 1y = 9
    // 1x + 3y = 7
    // Solution: x=4, y=1
    mtx::Matrix<double> A(2, 2);
    A.at(0, 0) = 2.0; A.at(0, 1) = 1.0;
    A.at(1, 0) = 1.0; A.at(1, 1) = 3.0;

    std::vector<double> b = {9.0, 7.0};

    std::vector<double> x = A.solve(b);
    
    ASSERT_NEAR(x[0], 4.0, 1e-9);
    ASSERT_NEAR(x[1], 1.0, 1e-9);

    // Test with the pivot case
    // 0x + 1y = 5
    // 2x + 1y = 7
    // Solution: y=5, x=1
    mtx::Matrix<double> B(2, 2);
    B.at(0, 0) = 0.0; B.at(0, 1) = 1.0;
    B.at(1, 0) = 2.0; B.at(1, 1) = 1.0;

    std::vector<double> b2 = {5.0, 7.0};
    
    std::vector<double> x2 = B.solve(b2);

    ASSERT_NEAR(x2[0], 1.0, 1e-9);
    ASSERT_NEAR(x2[1], 5.0, 1e-9);
}

mtx::Matrix<double> create_identity(size_t n) {
    mtx::Matrix<double> I(n, n);
    for (size_t i = 0; i < n; ++i) {
        I.at(i, i) = 1.0;
    }
    return I;
}

TEST(LinalgTest, Inverse) {
    // A = | 2  1 |
    //     | 1  3 |
    mtx::Matrix<double> A(2, 2);
    A.at(0, 0) = 2.0; A.at(0, 1) = 1.0;
    A.at(1, 0) = 1.0; A.at(1, 1) = 3.0;

    // --- Execute ---
    mtx::Matrix<double> A_inv = A.inverse();
    
    // Check A * A_inv
    // This uses your super-fast operator* !
    mtx::Matrix<double> C = A * A_inv;
    
    // C should be the Identity matrix
    mtx::Matrix<double> I = create_identity(2);

    // --- Assert ---
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(C.at(r, c), I.at(r, c), 1e-9);
        }
    }

    // Test a bigger matrix
    mtx::Matrix<double> B(3, 3);
    B.at(0, 0) = 3; B.at(0, 1) = 8; B.at(0, 2) = 4;
    B.at(1, 0) = 2; B.at(1, 1) = 1; B.at(1, 2) = 1;
    B.at(2, 0) = 5; B.at(2, 1) = 2; B.at(2, 2) = 7;

    mtx::Matrix<double> B_inv = B.inverse();
    mtx::Matrix<double> C2 = B * B_inv;
    mtx::Matrix<double> I3 = create_identity(3);

    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            ASSERT_NEAR(C2.at(r, c), I3.at(r, c), 1e-9);
        }
    }
}