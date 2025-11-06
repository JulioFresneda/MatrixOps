#pragma once
#include "MatrixExpression.hpp"

namespace mtx {

// Wraps an expression and adds a scalar to every element.
template<typename E, typename T>
class MatrixScalarAdd : public MatrixExpression<MatrixScalarAdd<E, T>, T> {
public:
    MatrixScalarAdd(const E& expr, T scalar) 
        : expr_(expr), scalar_(scalar) {}

    T at(size_t r, size_t c) const {
        return expr_.at(r, c) + scalar_;
    }

    size_t rows() const { return expr_.rows(); }
    size_t cols() const { return expr_.cols(); }

private:
    const E& expr_;
    T scalar_;
};

} 