#include <gtest/gtest.h>
#include "matrix/Matrix.hpp" 

// A simple "round-trip" test
TEST(IOTest, BinaryRoundTrip) {

    // 1. Save a random matrix
    mtx::Matrix<double> m1(20, 30);
    m1.fill_random(0.0, 1.0);
    m1.save_binary("test_matrix.bin");

    // 2. Load it back into a new matrix
    mtx::Matrix<double> m2("test_matrix.bin");

    // 3. Check if they are identical
    ASSERT_EQ(m1.rows(), m2.rows());
    ASSERT_EQ(m1.cols(), m2.cols());

    for (size_t r = 0; r < m1.rows(); ++r) {
        for (size_t c = 0; c < m1.cols(); ++c) {
            // Check that every single element is the same!
            ASSERT_DOUBLE_EQ(m1.at(r, c), m2.at(r, c));
        }
    }
}

// Add this to tests/test_io.cpp

TEST(IOTest, CSVRoundTrip) {
    // 1. Create a matrix and save it
    mtx::Matrix<double> m1(25, 15);
    m1.fill_random(0.0, 1.0);
    m1.save_csv("test_matrix.csv");

    // 2. Load it back into a new matrix
    mtx::Matrix<double> m2("test_matrix.csv", ',');

    // 3. Check dimensions
    ASSERT_EQ(m1.rows(), m2.rows());
    ASSERT_EQ(m1.cols(), m2.cols());

    // 4. Check values
    for (size_t r = 0; r < m1.rows(); ++r) {
        for (size_t c = 0; c < m1.cols(); ++c) {
            // Use ASSERT_NEAR for floating-point values
            // to account for tiny precision errors during text conversion
            ASSERT_NEAR(m1.at(r, c), m2.at(r, c), 1e-9); 
        }
    }
}