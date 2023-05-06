/*
 *  operators.h
 *  Created by Matthias Kesenheimer on 19.06.22.
 *  Copyright 2022. All rights reserved.
 *  More information about the Eigen library at http://eigen.tuxfamily.org/dox/index.html
 */
#pragma once
#include "vector.h"
#include "matrix.h"
#include <iostream>

namespace math {
    /// <summary>
    /// ostream
    /// </summary>
    template <class _T>
    std::ostream& operator<< (std::ostream& stream, const matrix<_T>& mat) {
        stream << mat.eigen();
        return stream;
    }

    /// <summary>
    /// ostream
    /// </summary>
    template <class _T>
    std::ostream& operator<< (std::ostream& stream, const vector<_T>& vec) {
        stream << vec.eigen();
        return stream;
    }

    namespace eigen {
        /// <summary>
        /// transpose a matrix
        /// </summary>
        template <class _T>
        inline matrix<_T> transpose(const matrix<_T>& mat) {
            return matrix<_T>(mat.eigen().transpose());
        }

        /// <summary>
        /// inverse of a matrix
        /// </summary>
        template <class _T>
        inline matrix<_T> inverse(const matrix<_T>& mat) {
            return matrix<_T>(mat.eigen().inverse());
        }

        /// <summary>
        /// general l-norm of a vector
        /// </summary>
        template <int _l, class _T>
        inline _T norm(const vector<_T>& vec) {
            return vec.eigen().template lpNorm<_l>();
        }

        /// <summary>
        /// norm of a vector
        /// </summary>
        template <class _T>
        inline _T norm(const vector<_T>& vec) {
            return vec.eigen().norm();
        }

        /// <summary>
        /// normalize a vector via general l-norm
        /// </summary>
        template <int _l, class _T>
        inline void normalize(vector<_T>& vec) {
            vec = vec / norm<_l, _T>(vec);
        }

        /// <summary>
        /// normalize a vector
        /// </summary>
        template <class _T>
        inline void normalize(vector<_T>& vec) {
            vec.eigen().normalize();
        }

        /// <summary>
        /// Frobenius norm of a matrix
        /// </summary>
        template <class _T>
        inline _T norm(const matrix<_T>& mat) {
            return mat.eigen().norm();
        }

        /// <summary>
        /// accumulate/sum all entries of a vector
        /// </summary>
        template <class _T>
        inline _T sum(const vector<_T>& vec) {
            return vec.eigen().sum();
        }
    }

    /// <summary>
    /// vector-scalar multiplication
    /// </summary>
    template <class _T>
    inline vector<_T> operator*(const vector<_T>& vec, const _T& scalar) {
        return vector<_T>(vec.eigen() * scalar);
    }

    /// <summary>
    /// vector-scalar multiplication
    /// </summary>
    template <class _T>
    inline vector<_T> operator*(const _T& scalar, const vector<_T>& vec) {
        return vector<_T>(scalar * vec.eigen());
    }

    /// <summary>
    /// vector-scalar division
    /// </summary>
    template <class _T>
    inline vector<_T> operator/(const vector<_T>& vec, const _T& scalar) {
        return vector<_T>(vec.eigen() / scalar);
    }

    /// <summary>
    /// vector-vector multiplication
    /// </summary>
    template <class _T>
    inline _T operator*(const vector<_T>& vecT, const vector<_T>& vec) {
        return static_cast<_T>(vecT.eigen().transpose() * vec.eigen());
    }

    namespace eigen {
        /// <summary>
        /// coefficient-wise vector multiplication: a[i] * b[i] = c[i]
        /// </summary>
        template <class _T>
        inline vector<_T> cprod(const vector<_T>& vec1, const vector<_T>& vec2) {
            return vector<_T>(vec1.eigen().cwiseProduct(vec2.eigen()));
            //return vector<_T>(vec1.eigen().array() * vec2.eigen().array());
        }

        /// <summary>
        /// coefficient-wise vector division: a[i] / b[i] = c[i]
        /// </summary>
        template <class _T>
        inline vector<_T> cdiv(const vector<_T>& vec1, const vector<_T>& vec2) {
            return vector<_T>(vec1.eigen().cwiseQuotient(vec2.eigen()));
            //return vector<_T>(vec1.eigen().array() * vec2.eigen().array());
        }
    }

    /// <summary>
    /// vector-vector addition
    /// </summary>
    template <class _T>
    inline vector<_T> operator+(const vector<_T>& lhs, const vector<_T>& rhs) {
        return vector<_T>(lhs.eigen() + rhs.eigen());
    }

    /// <summary>
    /// vector-vector addition
    /// </summary>
    template <class _T>
    inline vector<_T>& operator+=(vector<_T>& lhs, const vector<_T>& rhs) {
        lhs.eigen() += rhs.eigen();
        return lhs;
    }

    /// <summary>
    /// vector-vector subtraction
    /// </summary>
    template <class _T>
    inline vector<_T>& operator-=(vector<_T>& lhs, const vector<_T>& rhs) {
        lhs.eigen() -= rhs.eigen();
        return lhs;
    }

    /// <summary>
    /// vector-scalar multiplication
    /// </summary>
    template <class _T>
    inline vector<_T>& operator*=(vector<_T>& lhs, const _T& rhs) {
        lhs.eigen() *= rhs;
        return lhs;
    }

    /// <summary>
    /// vector-scalar division
    /// </summary>
    template <class _T>
    inline vector<_T>& operator/=(vector<_T>& lhs, const _T& rhs) {
        lhs.eigen() /= rhs;
        return lhs;
    }

    /// <summary>
    /// vector-vector subtraction
    /// </summary>
    template <class _T>
    inline typename vector<_T>::eigen_type operator-(const vector<_T>& lhs, const vector<_T>& rhs) {
        return lhs.eigen() - rhs.eigen();
    }

    /// <summary>
    /// matrix-scalar multiplication
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator*(const matrix<_T>& mat, const _T& scalar) {
        return mat.eigen() * scalar;
    }

    /// <summary>
    /// matrix-scalar multiplication
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator*(const _T& scalar, const matrix<_T>& mat) {
        return scalar * mat.eigen();
    }

    /// <summary>
    /// matrix-scalar division
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator/(const matrix<_T>& mat, const _T& scalar) {
        return mat.eigen() / scalar;
    }

    /// <summary>
    /// matrix-vector multiplication
    /// </summary>
    template <class _T>
    inline typename vector<_T>::eigen_type operator*(const matrix<_T>& mat, const vector<_T>& vec) {
        return mat.eigen() * vec.eigen();
    }

    /// <summary>
    /// vector-matrix multiplication
    /// </summary>
    template <class _T>
    inline typename vector<_T>::eigen_type operator*(const vector<_T>& vecT, const matrix<_T>& mat) {
        return vecT.eigen().transpose() * mat.eigen();
    }

    /// <summary>
    /// matrix-matrix multiplication
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator*(const matrix<_T>& lhs, const matrix<_T>& rhs) {
        return lhs.eigen() * rhs.eigen();
    }

    /// <summary>
    /// matrix-matrix addition
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator+(const matrix<_T>& lhs, const matrix<_T>& rhs) {
        return lhs.eigen() + rhs.eigen();
    }

    /// <summary>
    /// matrix-matrix subtraction
    /// </summary>
    template <class _T>
    inline typename matrix<_T>::eigen_type operator-(const matrix<_T>& lhs, const matrix<_T>& rhs) {
        return lhs.eigen() - rhs.eigen();
    }
}