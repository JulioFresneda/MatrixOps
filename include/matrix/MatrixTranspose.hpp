#pragma once
#include "MatrixExpression.hpp"

namespace mtx {

template<typename E, typename T>
class MatrixTranspose : public MatrixExpression<MatrixTranspose<E, T>, T> {
public:
    // Store a const reference to the expression we are transposing
    MatrixTranspose(const E& expr) : expr_(expr) {}

    // When asked for (r, c), we return the original's (c, r) <- Memory efficient
    T at(size_t r, size_t c) const {
        return expr_.at(c, r);
    }

    // The rows of the transpose are the columns of the original
    size_t rows() const { return expr_.cols(); }

    // The columns of the transpose are the rows of the original
    size_t cols() const { return expr_.rows(); }

private:
    const E& expr_;
};

} // namespace mtx