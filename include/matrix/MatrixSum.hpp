#pragma once
#include "MatrixExpression.hpp"

namespace mtx {
template<typename E1, typename E2, typename T>
class MatrixSum : public MatrixExpression<MatrixSum<E1, E2, T>, T> {
public:

    MatrixSum(const E1& lhs, const E2& rhs) 
        : lhs_(lhs), rhs_(rhs) 
    {
        
    }

    T at(size_t r, size_t c) const {
        return lhs_.at(r, c) + rhs_.at(r, c);
    }

    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return lhs_.cols(); }

private:
    const E1& lhs_;
    const E2& rhs_;
};

} // namespace mtx