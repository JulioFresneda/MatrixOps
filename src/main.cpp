#include "matrix/Matrix.hpp"
#include <iostream>

int main() {
    mtx::Matrix<int> A(5, 5);
    A.fill_random(0, 5); // Fill with smaller numbers
    A.print("Matrix A"); // <-- Print A

    mtx::Matrix<int> B(5, 5);
    B.fill_random(0, 5);
    B.print("Matrix B"); // <-- Print B

    std::cout << "Creating C = A + B..." << std::endl;
    mtx::Matrix<int> C = A + B;
    C.print("Matrix C (A + B)"); // <-- Print C

    std::cout << "\nCreating D = A + B + C..." << std::endl;
    mtx::Matrix<int> D = A + B + C;
    D.print("Matrix D (A + B + C)"); // <-- Print D

    std::cout << "\nAll done." << std::endl;
    
    return 0;
}