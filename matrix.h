/*
 *  matrix.h
 *  Created by Matthias Kesenheimer on 19.06.22.
 *  Copyright 2022. All rights reserved.
 *  More information about the Eigen library at http://eigen.tuxfamily.org/dox/index.html
 */

#pragma once
#include "vector.h"
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <initializer_list>
#include <vector>
#include <atomic>

//#define _DEBUG
#ifdef _DEBUG
#include <iostream>
#endif

namespace math {
    /// <summary>
    /// matrix class
    /// Verknuepft die Daten gespeichert in m_data mit Eigen::matrix. Dadurch koennen Rechenoperationen einfacher und schneller durchgefuehrt werden.
    /// </summary>
    template <class _T> 
    class matrix {
    public:
        /// <summary>
        /// typedefs
        /// </summary>
        using eigen_type = Eigen::Matrix<_T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using vector_type = Eigen::Matrix<_T, Eigen::Dynamic, 1>;
        //using map_type = Eigen::Map<eigen_type>;
        using vector_map_type = Eigen::Map<vector_type>;
        using value_type = typename eigen_type::value_type;
        using size_type = size_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = typename vector_map_type::iterator;
        using const_iterator = typename vector_map_type::const_iterator;
        using reverse_iterator = typename std::reverse_iterator<iterator>;
        using const_reverse_iterator = typename std::reverse_iterator<const_iterator>;
        
        /// <summary>
        /// construct a dynamic-size empty matrix
        /// </summary>
        matrix()
            : m_eigen(0, 0), m_refCount(1), m_reserved_memory_left(0) {}
        
        /// <summary>
        /// construct a dynamic-size matrix with size 'rows'x'cols'
        /// </summary>
        matrix(size_type r, size_type c)
            : m_eigen(eigen_type::Zero(r, c)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct a dynamic-size matrix with number of rows 'rows' and default column vectors 'vector'
        /// </summary>
        matrix(size_type r, const std::vector<_T>& v)
            : m_eigen(matrixFromVector(r, v)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct a dynamic-size matrix with number of rows 'rows' and default column vectors 'vector'
        /// </summary>
        matrix(size_type r, const vector<_T>& v)
            : m_eigen(matrixFromVector(r, v)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct a dynamic-size matrix with size 'cols * rows' and default value 'defaultValue'
        /// </summary>
        matrix(size_type r, size_type c, const value_type& defaultValue)
            : m_eigen(eigen_type::Constant(r, c, defaultValue)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct a dynamic-size matrix from nested vector's
        /// </summary>
        matrix(const vector<vector<_T>>& mat)
            : m_eigen(matrixFromVectors(mat)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// initialization by initializer list
        /// </summary>
        matrix(std::initializer_list<std::initializer_list<_T>> IList)
            : m_eigen(matrixFromVectors(IList)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// initialization by initializer list
        /// </summary>
        matrix(std::initializer_list<vector<_T>> IList)
            : m_eigen(matrixFromVectors(IList)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// copy constructor
        /// </summary>
        matrix(const matrix& other)
            : m_eigen(other.m_eigen), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// move constructor
        /// </summary>
        matrix(matrix&& other)
            : m_eigen(std::move(other.m_eigen)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct from eigen type
        /// </summary>
        matrix(const eigen_type& eigenmat)
            : m_eigen(eigenmat), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// construct from eigen type
        /// </summary>
        matrix(eigen_type&& eigenmat)
            : m_eigen(std::move(eigenmat)), m_refCount(1), m_reserved_memory_left(0) {}

        /// <summary>
        /// number of rows of the matrix
        /// </summary>
        size_type rows() const {
            return m_eigen.rows() - m_reserved_memory_left;
        }

        /// <summary>
        /// number of columns of the matrix
        /// </summary>
        size_type cols() const {
            return m_eigen.cols();
        }

        /// <summary>
        /// total size
        /// </summary>
        size_type size() const {
            return cols() * rows();
        }

        /// <summary>
        /// returns the underlying data structure
        /// </summary>
        value_type* data() {
            return m_eigen.data();
        }

        /// <summary>
        /// returns the underlying data structure
        /// </summary>
        const value_type* data() const {
            return m_eigen.data();
        }

        /// <summary>
        /// returns the underlying data structure
        /// </summary>
        eigen_type& eigen() {
            return m_eigen;
        }

        /// <summary>
        /// returns the underlying data structure
        /// </summary>
        const eigen_type& eigen() const {
            return m_eigen;
        }

        /// <summary>
        /// give back the matrix as a reshaped vector 
        /// </summary>
        const vector_map_type reshaped() const {
            return vector_map_type(m_eigen.data(), cols(), 1);
        }

        /// <summary>
        /// give back the matrix as a reshaped vector
        /// </summary>
        vector_map_type reshaped() {
            //return m_eigen.template reshaped<Eigen::RowMajor>(size(), 1);
            return vector_map_type(m_eigen.data(), cols(), 1);
        }

        /// <summary>
        /// begin of data container
        /// returns a const iterator
        /// </summary>
        const_iterator begin() const {
            return reshaped().begin();
        }

        /// <summary>
        /// begin of data container
        /// returns an iterator
        /// </summary>
        iterator begin() {
            return reshaped().begin();
        }

        /// <summary>
        /// end of data container
        /// returns a const iterator
        /// </summary>
        const_iterator end() const {
            return reshaped().end();
        }

        /// <summary>
        /// end of data container
        /// returns an iterator
        /// </summary>
        iterator end() {
            return reshaped().end();
        }

        /// <summary>
        /// rbegin of data container
        /// returns a const iterator
        /// </summary>
        const_reverse_iterator rbegin() const {
            return std::reverse_iterator(reshaped().end());
        }

        /// <summary>
        /// rbegin of data container
        /// returns an iterator
        /// </summary>
        reverse_iterator rbegin() {
            return std::reverse_iterator(reshaped().end());
        }

        /// <summary>
        /// rend of data container
        /// returns a const iterator
        /// </summary>
        const_reverse_iterator rend() const {
            return std::reverse_iterator(reshaped().begin());
        }

        /// <summary>
        /// rend of data container
        /// returns an iterator
        /// </summary>
        reverse_iterator rend() {
            return std::reverse_iterator(reshaped().begin());
        }

        /// <summary>
        /// is the matrix empty?
        /// </summary>
        const bool empty() const {
            return size() == 0;
        }

        /// <summary>
        /// add new row vector
        /// </summary>
        template <class _Vec>
        void push_back(const _Vec& vec) {
            size_type cols = m_eigen.cols();
            if (m_eigen.cols() == 0)
                cols = vec.size();

#if defined(_DEBUG) || defined(DEBUG)
            if (vector.size() != m_eigen.cols()) {
                std::cout << "Warning: matrix::push_back: size of vector does not match the matrix layout." << std::endl;
                std::cout << "vector.size() = " << vector.size() << ", matrix.cols() = " << m_eigen.cols() << std::endl;
            }
#endif

            if (m_reserved_memory_left > 0) {
                for (size_type c = 0; c < m_eigen.cols(); ++c)
                    m_eigen(rows(), c) = vec[c];
                m_reserved_memory_left--;
                return;
            }

            m_eigen.conservativeResize(m_eigen.rows() + 1, cols);
            for (size_type c = 0; c < vec.size(); ++c)
                m_eigen(m_eigen.rows() - 1, c) = vec[c];
        }

        /// <summary>
        /// add new row vector
        /// </summary>
        template <class _Vec>
        void push_back(_Vec&& vec) {
            size_type cols = m_eigen.cols();
            if (m_eigen.cols() == 0)
                cols = vec.size();

#if defined(_DEBUG) || defined(DEBUG)
            if (vector.size() != m_eigen.cols()) {
                std::cout << "Warning: matrix::push_back: size of vector does not match the matrix layout." << std::endl;
                std::cout << "vector.size() = " << vector.size() << ", matrix.cols() = " << m_eigen.cols() << std::endl;
            }
#endif

            if (m_reserved_memory_left > 0) {
                for (size_type c = 0; c < m_eigen.cols(); ++c)
                    m_eigen(rows(), c) = std::move(vec[c]);
                m_reserved_memory_left--;
                return;
            }

            m_eigen.conservativeResize(m_eigen.rows() + 1, cols);
            for (size_type c = 0; c < vec.size(); ++c)
                m_eigen(m_eigen.rows() - 1, c) = std::move(vec[c]);
        }

        /// <summary>
        /// add new row vector by initializer list
        /// </summary>
        void push_back(std::initializer_list<_T> vec) {
            size_type cols = m_eigen.cols();
            if (m_eigen.cols() == 0)
                cols = vec.size();

#if defined(_DEBUG) || defined(DEBUG)
            if (vector.size() != m_eigen.cols()) {
                std::cout << "Warning: matrix::push_back: size of vector does not match the matrix layout." << std::endl;
                std::cout << "vector.size() = " << vector.size() << ", matrix.cols() = " << m_eigen.cols() << std::endl;
            }
#endif

            if (m_reserved_memory_left > 0) {
                size_type c = 0;
                for (const auto& value : vec)
                    m_eigen(rows(), c++) = value;
                m_reserved_memory_left--;
                return;
            }

            m_eigen.conservativeResize(m_eigen.rows() + 1, cols);
            size_type c = 0;
            for (const auto& value : vec)
                m_eigen(m_eigen.rows() - 1, c) = value;
        }

        /// <summary>
        /// reserve memory for pushing back row vectors
        /// (does not change size, but subsequent push_back()'s are more efficient)
        /// Note: only rows can be reserved, since a push_back() does only increase rows and not columns.
        /// If matrix is empty, reserve rows as well as columns.
        /// </summary>
        void reserve_rows(size_type r, size_type c) {
            if ((m_eigen.rows() == 0 && m_eigen.cols() == 0) || (r > m_eigen.rows() && c == m_eigen.cols())) {
                m_reserved_memory_left = r - m_eigen.rows();
                m_eigen.conservativeResize(r, c);
            }
        }
        void reserve_rows(size_type r) {
#if defined(_DEBUG) || defined(DEBUG)
            if (m_eigen.cols() == 0)
                std::cout << "Error: Matrix empty. Use reserve_rows(size_type r, size_type c) instead." << std::endl;
#endif
            if (r > m_eigen.rows()) {
                m_reserved_memory_left = r - m_eigen.rows();
                m_eigen.conservativeResize(r, m_eigen.cols());
            }
        }

        /// <summary>
        /// delete entries, leaving the container with a size of 0.
        /// </summary>
        void clear() {
            m_eigen = eigen_type(0, 0);
        }

        /// <summary>
        /// clear without change of size
        /// </summary>
        void reset() {
            m_eigen = eigen_type::Zero(m_eigen.rows(), m_eigen.cols());
        }

        /// <summary>
        /// assign a new size and new values to the vector
        /// </summary>
        void assign(size_type r, size_type c, const value_type& defaultValue) {
            m_eigen = eigen_type::Constant(r, c, defaultValue);
        }

        /// <summary>
        /// resize, conserve the values
        /// </summary>
        void resize(size_type r, size_type c) {
            m_eigen.resize(r, c);
        }

        /// <summary>
        /// accessing elements
        /// </summary>
        value_type& at(const size_type r, const size_type c) {
            return m_eigen(r, c);
        }

        /// <summary>
        /// accessing elements
        /// </summary>
        const value_type& at(const size_type r, const size_type c) const {
            return m_eigen(r, c);
        }

        /// <summary>
        /// bracket operator
        /// </summary>
        value_type& operator()(const size_type r, const size_type c) {
            return m_eigen(r, c);
        }

        /// <summary>
        /// bracket operator
        /// </summary>
        const value_type& operator()(const size_type r, const size_type c) const {
            return m_eigen(r, c);
        }

        /// <summary>
        /// bracket operator
        /// </summary>
        vector_map_type operator[](const size_type r) {
            value_type* ptr = m_eigen.data() + r * cols();
            return vector_map_type(ptr, cols(), 1);
        }

        /// <summary>
        /// bracket operator
        /// </summary>
        const vector_map_type operator[](const size_type r) const {
            value_type* ptr = m_eigen.data() + r * cols();
            return vector_map_type(ptr, cols(), 1);
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        const matrix<_T>& operator=(const matrix<_T>& rhs) {
            if (this != &rhs)
                m_eigen = rhs.m_eigen;
            return *this;
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        matrix<_T>& operator=(matrix<_T>&& rhs) {
            if (this != &rhs)
                m_eigen = std::move(rhs.m_eigen);
            return *this;
        }

        /// <summary>
        /// increase the reference count
        /// </summary>
        void addRef() {
            ++m_refCount;
        }

        /// <summary>
        /// decrese the reference count
        /// </summary>
        void release() {
            if (!--m_refCount)
                delete this;
        }

    private:
        /// <summary>
        /// generate a matrix from a single vector
        /// </summary>
        template <class _Vec>
        const eigen_type matrixFromVector(size_type rows, const _Vec& v) {
            eigen_type mat(rows, v.size());
            for (size_type r = 0; r < rows; ++r)
                for (size_type c = 0; c < v.size(); ++c)
                    mat(r, c) = v[c];
            return mat;
        }

        /// <summary>
        /// generate a matrix from multiple vectors
        /// </summary>
        template <class _Vec>
        const eigen_type matrixFromVectors(const std::initializer_list<_Vec>& list) {
#ifdef _DEBUG
            size_type cols = (list.size() > 0 ? list.begin()->size() : 0);
#else
            size_type cols = list.begin()->size();
#endif
            eigen_type mat(list.size(), cols);
            size_type it = 0;
            for (const auto& vec : list) // rows
                for (const auto& value : vec) // columns
                    mat.data()[it++] = value;
            return mat;
        }

        /// <summary>
        /// generate a matrix from multiple vectors
        /// </summary>
        const eigen_type matrixFromVectors(const vector<vector<_T>>& mat) {
#ifdef _DEBUG
            size_type cols = (mat.size() > 0 ? mat.begin()->size() : 0);
#else
            size_type cols = mat.front().size();
#endif
            eigen_type matr(mat.size(), cols);
            size_type it = 0;
            for (const auto& vec : mat) // rows
                for (const auto& value : vec) // columns
                    matr.data()[it++] = value;
            return matr;
        }

        /// <summary>
        /// private/underlying data structure
        /// </summary>
        eigen_type m_eigen;
        std::atomic<int> m_refCount;
        size_type m_reserved_memory_left;
    };
}