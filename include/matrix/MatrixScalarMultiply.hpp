#pragma once
#include "MatrixExpression.hpp"

namespace mtx {

// This expression wraps another expression and multiplies
// every element by a scalar value.
template<typename E, typename T>
class MatrixScalarMultiply : public MatrixExpression<MatrixScalarMultiply<E, T>, T> {
public:
    // Store the expression and the scalar value
    MatrixScalarMultiply(const E& expr, T scalar) 
        : expr_(expr), scalar_(scalar) {}

    // When asked for (r, c), we return the original's (r, c) * scalar
    T at(size_t r, size_t c) const {
        return expr_.at(r, c) * scalar_;
    }

    // Dimensions are unchanged
    size_t rows() const { return expr_.rows(); }
    size_t cols() const { return expr_.cols(); }

private:
    const E& expr_;
    T scalar_;
};

}