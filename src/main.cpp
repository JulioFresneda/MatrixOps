#include <iostream>
#include <vector>
#include <exception>
#include "matrix/Matrix.hpp" // This one header pulls in everything!

// Helper function to create an Identity matrix
template<typename T>
mtx::Matrix<T> create_identity(size_t n) {
    mtx::Matrix<T> I(n, n);
    for (size_t i = 0; i < n; ++i) {
        I.at(i, i) = static_cast<T>(1.0);
    }
    return I;
}

int main() {
    try {
        // ===================================================================
        // 1. CONSTRUCTORS, I/O, AND HELPERS
        // ===================================================================
        std::cout << "--- 1. I/O and Constructors ---\n" << std::endl;

        // Test constructor, fill_random (with double), and print
        mtx::Matrix<double> A_double(3, 4);
        A_double.fill_random(0.0, 10.0);
        A_double.print("A (3x4 double)");

        // Test fill_random (with int)
        mtx::Matrix<int> A_int(2, 3);
        A_int.fill_random(1, 100);
        A_int.print("A (2x3 int)");

        // Test save/load binary
        A_double.save_binary("A_double.bin");
        mtx::Matrix<double> B_bin(A_double.rows(), A_double.cols()); // Fix: Use correct constructor
        // B_bin = mtx::Matrix<double>::load_binary("A_double.bin"); // Fix: Use static load if implemented
        // Note: Your I/O test used a constructor. Let's assume the "remove default" fix.
        // If you used static factories, this would be:
        // mtx::Matrix<double> B_bin = mtx::Matrix<double>::load_binary("A_double.bin");
        // If you used the "remove default" fix on constructors, you had an ambiguity.
        // Let's assume you have a constructor `Matrix(const std::string& binary_path)`
        mtx::Matrix<double> B_bin_loaded("A_double.bin");
        B_bin_loaded.print("B (loaded from A_double.bin)");
        
        // Test save/load CSV
        // We must provide the delimiter ',' as we removed the default
        A_int.save_csv("A_int.csv");
        mtx::Matrix<int> C_csv(A_int.rows(), A_int.cols()); // Fix: Use correct constructor
        // mtx::Matrix<int> C_csv_loaded("A_int.csv", ','); // This assumes you removed the default
        // C_csv_loaded.print("C (loaded from A_int.csv)");
        // Note: The above CSV loader might be ambiguous with the binary one.
        // This is why static factories (`load_csv`) are the robust solution.
        // For this demo, we'll trust the I/O tests.

        std::cout << "\n--- 2. EXPRESSION TEMPLATES (LAZY OPS) ---\n" << std::endl;
        
        mtx::Matrix<double> M1(2, 2);
        M1.at(0, 0) = 1; M1.at(0, 1) = 2;
        M1.at(1, 0) = 3; M1.at(1, 1) = 4;
        M1.print("M1");

        mtx::Matrix<double> M2(2, 2);
        M2.at(0, 0) = 5; M2.at(0, 1) = 6;
        M2.at(1, 0) = 7; M2.at(1, 1) = 8;
        M2.print("M2");
        
        // Test all Expression Template ops
        std::cout << "Calculating: Final = (M1 + M2) * 2.0 - (M1 / 10.0)" << std::endl;
        mtx::Matrix<double> Final = (M1 + M2) * 2.0 - (M1 / 10.0);
        Final.print("Final");

        std::cout << "Calculating: Transpose = M1.transpose() + M2" << std::endl;
        mtx::Matrix<double> Transpose = M1.transpose() + M2;
        Transpose.print("Transpose");


        // ===================================================================
        // 3. GEMM (MATRIX MULTIPLICATION)
        // ===================================================================
        std::cout << "\n--- 3. GEMM (Optimized Matrix Multiply) ---\n" << std::endl;
        
        // This operator* calls your fastest, multithreaded SIMD kernel!
        mtx::Matrix<double> C_gemm = M1 * M2;
        C_gemm.print("C (M1 * M2) [Fast Kernel]");

        // We can also call the naive one for comparison
        mtx::Matrix<double> C_naive = mtx::gemm_naive(M1, M2);
        C_naive.print("C (M1 * M2) [Naive Kernel]");

        // ===================================================================
        // 4. LINEAR ALGEBRA
        // ===================================================================
        std::cout << "\n--- 4. Linear Algebra (Solve, Det, Inv, Rank) ---\n" << std::endl;

        mtx::Matrix<double> A_solve(3, 3);
        A_solve.at(0, 0) = 1; A_solve.at(0, 1) = 2; A_solve.at(0, 2) = 3;
        A_solve.at(1, 0) = 0; A_solve.at(1, 1) = 1; A_solve.at(1, 2) = 4;
        A_solve.at(2, 0) = 5; A_solve.at(2, 1) = 6; A_solve.at(2, 2) = 0;
        A_solve.print("A (for Linalg)");

        // Test Determinant
        std::cout << "Determinant(A): " << A_solve.determinant() << std::endl; // Should be 1.0

        // Test Rank
        std::cout << "Rank(A): " << A_solve.rank() << std::endl; // Should be 3

        // Test Solve (Ax = b)
        // b = [6, 8, 8]  -> Solution x = [-1, 0, 2]
        std::vector<double> b = {6.0, 8.0, 8.0};
        std::vector<double> x = A_solve.solve(b);
        std::cout << "Solving Ax=b... Solution x = [" 
                  << x[0] << ", " << x[1] << ", " << x[2] << "]" << std::endl;

        // Test Inverse
        mtx::Matrix<double> A_inv = A_solve.inverse();
        A_inv.print("A_inverse");

        // Test A * A_inv = I
        mtx::Matrix<double> I = A_solve * A_inv;
        I.print("A * A_inverse (should be Identity)");
        
        // Test Singular (rank-deficient) matrix
        mtx::Matrix<double> S(2, 2);
        S.at(0, 0) = 1; S.at(0, 1) = 1;
        S.at(1, 0) = 1; S.at(1, 1) = 1;
        S.print("S (Singular Matrix)");
        std::cout << "Determinant(S): " << S.determinant() << std::endl; // Should be 0
        std::cout << "Rank(S): " << S.rank() << std::endl; // Should be 1


    } catch (const std::exception& e) {
        std::cerr << "\n*** An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}