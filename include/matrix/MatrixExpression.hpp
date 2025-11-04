#pragma once
#include <cstddef> // for size_t

namespace mtx {
template<typename E, typename T>

// ########################## What is this? ##########################
// ---------------------------------------------------------------------------
// An expression is something like M = A + B + C
// Usually, the compiler calculates first B + C and stores it in a temp value.
// Then, it calculates A + temp and stores it in M. 
// This is inefficient. With this class, we can calculate all at once.
// ---------------------------------------------------------------------------

class MatrixExpression {
public:
    T at(size_t r, size_t c) const {
        return static_cast<const E*>(this)->at(r, c);
    }

    size_t rows() const {
        return static_cast<const E*>(this)->rows();
    }

    size_t cols() const {
        return static_cast<const E*>(this)->cols();
    }
};

} // namespace mtx