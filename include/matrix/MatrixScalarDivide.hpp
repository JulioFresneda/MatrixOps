#pragma once
#include "MatrixExpression.hpp"
#include <stdexcept>

namespace mtx {

// Wraps an expression and divides every element by a scalar.
template<typename E, typename T>
class MatrixScalarDivide : public MatrixExpression<MatrixScalarDivide<E, T>, T> {
public:
    MatrixScalarDivide(const E& expr, T scalar) 
        : expr_(expr), scalar_(scalar) {
        if (scalar == T(0)) {
            throw std::runtime_error("Division by zero: scalar cannot be zero.");
        }
    }
    T at(size_t r, size_t c) const {
        return expr_.at(r, c) / scalar_;
    }

    size_t rows() const { return expr_.rows(); }
    size_t cols() const { return expr_.cols(); }

private:
    const E& expr_;
    T scalar_;
};

} 