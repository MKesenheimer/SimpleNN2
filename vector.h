/*
 *  vector.h
 *  Created by Matthias Kesenheimer on 19.06.22.
 *  Copyright 2022. All rights reserved.
 *  More information about the Eigen library at http://eigen.tuxfamily.org/dox/index.html
 */

#pragma once
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <initializer_list>
#include <atomic>

#define _DEBUG
#ifdef _DEBUG
#include <iostream>
#endif

namespace math {
    /// <summary>
    /// vector class
    /// Verknuepft die Daten gespeichert in m_data mit Eigen::matrix. Dadurch koennen Rechenoperationen einfacher und schneller durchgefuehrt werden.
    /// </summary>
    template <class _T>
    class vector {
    public:
        /// <summary>
        /// typedefs
        /// </summary>
        using eigen_type = Eigen::Matrix<_T, Eigen::Dynamic, 1>;
        using map_type = Eigen::Map<eigen_type>;
        using value_type = typename eigen_type::value_type;
        using size_type = size_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = typename eigen_type::iterator;
        using const_iterator = typename eigen_type::const_iterator;
        using reverse_iterator = typename std::reverse_iterator<iterator>;
        using const_reverse_iterator = typename std::reverse_iterator<const_iterator>;

        // Notiz zum Zaehlen der Referenzen:
        // - Beim Anlegen eines Objekts muss der Referenzzaehler immer 1 sein, d.h. auch wenn der move-constructor 
        //   vector(vector&& other) verwendet wird. Es wird ein neues Objekt erzeugt, auf die bisher nur eine Referenz zeigt.
        //   Es spielt keine Rolle, ob das Objekt durch eine Move-Operation oder durch den Standardkonstruktor erzeugt wird.
        //   Beim Move-Konstruktor sollen ja nur die Daten verschoben werden. Die Referenzen zum alten Objekt sollen trotzdem bestehen
        //   bleiben.
        // - Bei den Kopier- bzw. Move-Operatoren const vector<_T>& operator=(const vector<_T>& rhs) und vector<_T>& operator=(const vector<_T>&& rhs)
        //   werden ausserdem auch nur die Daten kopiert bzw. verschoben. Die urspruenglichen Referenzen duerfen von dieser Aktion nicht beeinflusst werden.
        //   Deshalb werden in diesen Faellen die Referenzzaehler nicht veraendert.

        /// <summary>
        /// construct a dynamic-size empty vector
        /// </summary>
        vector()
            : m_eigen(0), m_refCount(1) {}

        /// <summary>
        /// construct a dynamic-size vector with size 'size'
        /// </summary>
        vector(size_type s)
            : m_eigen(s), m_refCount(1) {}

        /// <summary>
        /// construct an object from an std::vector
        /// </summary>
        vector(const std::vector<_T>& v)
            : m_eigen(v.size()), m_refCount(1) {
                for (size_type i = 0; i < v.size(); ++i)
                    m_eigen[i] = v[i];
            }

        /// <summary>
        /// construct from plain array
        /// </summary>
        vector(const value_type* v, size_type s)
            : m_eigen(s), m_refCount(1) {
                for (size_type i = 0; i < s; ++i)
                    m_eigen[i] = v[i];
            }

        /// <summary>
        /// construct a dynamic-size vector with size 'size' and default value 'defaultValue'
        /// </summary>
        vector(size_type s, const value_type& defaultValue)
            : m_eigen(s), m_refCount(1) {
                for (size_type i = 0; i < s; ++i)
                    m_eigen[i] = defaultValue;
            }

        /// <summary>
        /// initializing by initializer list
        /// </summary>
        vector(std::initializer_list<value_type> l)
            : m_eigen(l.size()), m_refCount(1) {
                for (size_type i = 0; i < l.size(); ++i)
                    m_eigen[i] = *(l.begin() + i);
            }

        /// <summary>
        /// construct from Eigen::matrix
        /// </summary>
        vector(const eigen_type& eigvec)
            :  m_eigen(eigvec), m_refCount(1) {}

        /// <summary>
        /// copy constructor
        /// </summary>
        vector(const vector& other)
            : m_eigen(other.m_eigen), m_refCount(1) {}

        /// <summary>
        /// move constructor
        /// </summary>
        vector(vector&& other) noexcept
            : m_eigen(std::move(other.m_eigen)), m_refCount(1) {}

        /// <summary>
        /// size of the data container
        /// </summary>
        const size_type size() const {
            return m_eigen.rows();
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
        /// begin of data container
        /// returns a const iterator
        /// </summary>
        const_iterator begin() const {
            return m_eigen.begin();
        }

        /// <summary>
        /// begin of data container
        /// returns an iterator
        /// </summary>
        iterator begin() {
            return m_eigen.begin();
        }

        /// <summary>
        /// end of data container
        /// returns a const iterator
        /// </summary>
        const_iterator end() const {
            return m_eigen.end();
        }

        /// <summary>
        /// end of data container
        /// returns an iterator
        /// </summary>
        iterator end() {
            return m_eigen.end();
        }

        /// <summary>
        /// rbegin of data container
        /// returns a const iterator
        /// </summary>
        const_reverse_iterator rbegin() const {
            return std::reverse_iterator(m_eigen.end());
        }

        /// <summary>
        /// rbegin of data container
        /// returns an iterator
        /// </summary>
        reverse_iterator rbegin() {
            return std::reverse_iterator(m_eigen.end());
        }

        /// <summary>
        /// rend of data container
        /// returns a const iterator
        /// </summary>
        const_reverse_iterator rend() const {
            return std::reverse_iterator(m_eigen.begin());
        }

        /// <summary>
        /// rend of data container
        /// returns an iterator
        /// </summary>
        reverse_iterator rend() {
            return std::reverse_iterator(m_eigen.begin());
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
        /// is the vector empty?
        /// </summary>
        const bool empty() const {
            return size() == 0;
        }

        /// <summary>
        /// add new value
        /// </summary>
        void push_back(const value_type& value) {
            //mat.conservativeResize(mat.rows(), mat.cols()+1);
            //mat.col(mat.cols() - 1) = vec;
            m_eigen.conservativeResize(m_eigen.rows() + 1, 1);
            m_eigen[m_eigen.rows() - 1] = value;
        }

        /// <summary>
        /// add new value
        /// </summary>
        void push_back(value_type&& value) {
            m_eigen.conservativeResize(m_eigen.rows() + 1, 1);
            m_eigen[m_eigen.rows() - 1] = std::move(value);
        }

        /// <summary>
        /// append a vector
        /// </summary>
        void append(const vector<value_type>& toAppend) {
            size_type old = m_eigen.rows();
            m_eigen.conservativeResize(m_eigen.rows() + toAppend.size(), 1);
            for (int i = 0; i < toAppend.size(); ++i)
                m_eigen[old + i] = toAppend[i];
        }

        /// <summary>
        /// append a vector
        /// </summary>
        void append(vector<value_type>&& toAppend) {
            size_type old = m_eigen.rows();
            m_eigen.conservativeResize(m_eigen.rows() + toAppend.size(), 1);
            for (int i = 0; i < toAppend.size(); ++i)
                m_eigen[old + i] = std::move(toAppend[i]);
        }

        /// <summary>
        /// delete entries, leaving the container with a size of 0.
        /// </summary>
        void clear() {
            //m_eigen.resize(0, 0);
            m_eigen = eigen_type(0);
        }

        /// <summary>
        /// clear without changing size
        /// </summary>
        void reset() {
            m_eigen = eigen_type(size());
            for (size_type i = 0; i < size(); ++i)
                    m_eigen[i] = 0;
        }

        /// <summary>
        /// assign a new size and new values to the vector
        /// </summary>
        void assign(size_type size, const value_type& defaultValue) {
            m_eigen = eigen_type(size);
            for (size_type i = 0; i < size; ++i)
                    m_eigen[i] = defaultValue;
        }

        /// <summary>
        /// resize the vector
        /// </summary>
        void resize(size_type newSize) {
            m_eigen.conservativeResize(newSize, 1);
        }

        /// <summary>
        /// reserve memory (does not change size, but subsequent push_back()'s are more efficient)
        /// </summary>
        void reserve(size_type size) {
            m_eigen = eigen_type(size);
        }

        /// <summary>
        /// erase with iterator
        /// </summary>
        void erase(iterator it) {
            size_type numRows = size() - 1;
            size_type numCols = 1;

            size_type rowToRemove = std::distance(begin(), it);
            if (rowToRemove < numRows)
                m_eigen.block(rowToRemove, 0, numRows - rowToRemove, numCols) = m_eigen.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

            m_eigen.conservativeResize(numRows, numCols);
        }

        /// <summary>
        /// erase elements in iterator range
        /// </summary>
        void erase(iterator it1, iterator it2) {
            size_type numberToRemove = std::distance(it1, it2);
            size_type numRows = size() - numberToRemove;
            size_type numCols = 1;

            size_type rowToRemove = std::distance(begin(), it1);
            if (rowToRemove < numRows)
                m_eigen.block(rowToRemove, 0, numRows - rowToRemove, numCols) = m_eigen.block(rowToRemove + numberToRemove, 0, numRows - rowToRemove, numCols);

            m_eigen.conservativeResize(numRows, numCols);
        }
        
        /// <summary>
        /// accessing elements
        /// </summary>
        const_reference operator[] (size_type const i) const {
            return m_eigen[i];
        }

        /// <summary>
        /// accessing elements
        /// </summary>
        reference operator[] (size_type const i) {
            return m_eigen[i];
        }

        /// <summary>
        /// accessing elements
        /// </summary>
        reference at(const size_type i) {
            return m_eigen(i);
        }

        /// <summary>
        /// accessing elements
        /// </summary>
        const_reference at(const size_type i) const {
            return m_eigen(i);
        }

        /// <summary>
        /// first element
        /// </summary>
        const_reference front() const {
            return m_eigen(0);
        }

        /// <summary>
        /// first element
        /// </summary>
        reference front() {
            return m_eigen(0);
        }

        /// <summary>
        /// last element
        /// </summary>
        const_reference back() const {
            return m_eigen(size() - 1);
        }

        /// <summary>
        /// last element
        /// </summary>
        reference back() {
            return m_eigen(size() - 1);
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        const vector<_T>& operator=(const vector<_T>& rhs) {
            if (this != &rhs) {
                m_eigen = rhs.m_eigen;
            }
            return *this;
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        vector<_T>& operator=(vector<_T>&& rhs) noexcept {
            if (this != &rhs) {
                m_eigen = std::move(rhs.m_eigen);
            }
            return *this;
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        const vector<_T>& operator=(const eigen_type& rhs) {
            // TODO: check size of 'this' vector
            m_eigen = rhs;
            return *this;
        }

        /// <summary>
        /// assignment operator
        /// </summary>
        vector<_T>& operator=(eigen_type&& rhs) noexcept {
            // TODO: check size of 'this' vector
            m_eigen = std::move(rhs);
            return *this;
        }

        /// <summary>
        /// add reference
        /// </summary>
        void addRef() {
            ++m_refCount;
        }

        /// <summary>
        /// remove reference
        /// </summary>
        void release() {
            if (!--m_refCount)
                delete this;
        }

        private:
            eigen_type m_eigen;
            std::atomic<int> m_refCount;
    };
}