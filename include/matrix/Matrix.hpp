#pragma once

#include <vector>
#include <iostream>
#include <fstream>   // For std::ofstream
#include <random>    // For std::mt19937
#include <stdexcept> // For std::runtime_error
#include <iomanip>   // For std::setprecision
#include <type_traits> // For std:is_floating_point_v
#include "MatrixSum.hpp"

namespace mtx {

template<typename T>
class Matrix : public MatrixExpression<Matrix<T>, T>{
public:

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


private:
    size_t num_rows;
    size_t num_cols;

    // We will use a 1D vector to store the 2D data internally.
    // We use size_t for extremely large matrices.
    std::vector<T> data; 
};

// ########################################################################
// -------------------------- Operator Overloads --------------------------
// ########################################################################
template<typename E1, typename E2, typename T>
auto operator+(const MatrixExpression<E1, T>& lhs, const MatrixExpression<E2, T>& rhs) {
    // This doesn't calculate anything!
    return mtx::MatrixSum<E1, E2, T>(
        *static_cast<const E1*>(&lhs), 
        *static_cast<const E2*>(&rhs)
    );
}


} // namespace mtx