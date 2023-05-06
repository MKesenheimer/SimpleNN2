#include <vector>
#include <iostream>
#include <gtest/gtest.h>
#include <string>

#include "vector.h"
#include "matrix.h"
#include "operators.h"

int add(int a, int b) {return a + b;}

TEST(Addition, CanAddTwoNumbers) {
  EXPECT_TRUE(add(2, 2) == 4);
}

using namespace math;
typedef vector<double> vectord;
typedef matrix<double> matrixd;

TEST(EigenArraysTest, TestVector1dConstructors) {
    // ctor with size
    vectord v1(3);
    EXPECT_EQ(3, v1.size());
    EXPECT_EQ(3, v1.eigen().size());

    // ctor with default values
    vectord v2(3, 1);
    EXPECT_EQ(3, v2.size());
    EXPECT_EQ(1, v2.data()[0]);
    EXPECT_EQ(1, v2.data()[1]);
    EXPECT_EQ(1, v2.data()[2]);

    EXPECT_EQ(3, v2.eigen().size());
    EXPECT_EQ(1, v2.eigen()[0]);
    EXPECT_EQ(1, v2.eigen()[1]);
    EXPECT_EQ(1, v2.eigen()[2]);

    v2.eigen()[0] = 2;
    EXPECT_EQ(2, v2.data()[0]);
    EXPECT_EQ(1, v2.data()[1]);
    EXPECT_EQ(1, v2.data()[2]);

    v2.data()[2] = 3;
    EXPECT_EQ(2, v2.eigen()[0]);
    EXPECT_EQ(1, v2.eigen()[1]);
    EXPECT_EQ(3, v2.eigen()[2]);

    // empty ctor
    vectord v3;
    EXPECT_EQ(0, v3.size());
    EXPECT_EQ(0, v3.eigen().size());

    // copy ctor
    vectord v4(v2);
    EXPECT_EQ(3, v2.size());
    EXPECT_EQ(2, v2.data()[0]);
    EXPECT_EQ(1, v2.data()[1]);
    EXPECT_EQ(3, v2.data()[2]);

    // initializer list
    vectord v5 = {1, 2, 3};
    EXPECT_EQ(3, v5.size());
    EXPECT_EQ(1, v5.data()[0]);
    EXPECT_EQ(2, v5.data()[1]);
    EXPECT_EQ(3, v5.data()[2]);
    EXPECT_EQ(1, v5.eigen()[0]);
    EXPECT_EQ(2, v5.eigen()[1]);
    EXPECT_EQ(3, v5.eigen()[2]);

    // with eigen matrix
    vectord v6(v5.eigen());
    EXPECT_EQ(3, v6.size());
    EXPECT_EQ(1, v6.data()[0]);
    EXPECT_EQ(2, v6.data()[1]);
    EXPECT_EQ(3, v6.data()[2]);
    EXPECT_EQ(1, v6.eigen()[0]);
    EXPECT_EQ(2, v6.eigen()[1]);
    EXPECT_EQ(3, v6.eigen()[2]);
}

TEST(EigenArraysTest, TestVector1dIterators) {
    vectord v1 = { 1, 2, 3 };
    int i = 0;
    for (const auto& e : v1)
        EXPECT_EQ(++i, e);

    vectord v2(3);
    i = 0;
    for (auto& e : v2)
        e = ++i;

    i = 0;
    for (const auto& e : v2)
        EXPECT_EQ(++i, e);
}

TEST(EigenArraysTest, TestVector1dPushBack)
{
    vectord v1 = { 1, 2 };
    v1.push_back(3);
    EXPECT_EQ(1, v1.data()[0]);
    EXPECT_EQ(2, v1.data()[1]);
    EXPECT_EQ(3, v1.data()[2]);
    EXPECT_EQ(1, v1.eigen()[0]);
    EXPECT_EQ(2, v1.eigen()[1]);
    EXPECT_EQ(3, v1.eigen()[2]);
}

TEST(EigenArraysTest, TestVector1dAppend)
{
    vectord v1 = { 1, 2, 3 };
    vectord v2 = { 4, 5, 6 };
    v1.append(v2);
    EXPECT_EQ(1, v1.data()[0]);
    EXPECT_EQ(2, v1.data()[1]);
    EXPECT_EQ(3, v1.data()[2]);
    EXPECT_EQ(4, v1.data()[3]);
    EXPECT_EQ(5, v1.data()[4]);
    EXPECT_EQ(6, v1.data()[5]);
    EXPECT_EQ(1, v1.eigen()[0]);
    EXPECT_EQ(2, v1.eigen()[1]);
    EXPECT_EQ(3, v1.eigen()[2]);
    EXPECT_EQ(4, v1.eigen()[3]);
    EXPECT_EQ(5, v1.eigen()[4]);
    EXPECT_EQ(6, v1.eigen()[5]);
}

TEST(EigenArraysTest, TestVector1dClear) {
    vectord v1 = { 1, 2, 3 };
    v1.clear();
    
    EXPECT_EQ(0, v1.size());
    EXPECT_EQ(0, v1.eigen().size());
}

TEST(EigenArraysTest, TestVector1dreset) {
    vectord v1 = { 1, 2, 3 };
    v1.reset();

    EXPECT_EQ(3, v1.size());
    EXPECT_EQ(3, v1.eigen().size());
}

TEST(EigenArraysTest, TestVector1dAssign)
{
    vectord v1 = { 6, 7, 8, 9 };
    v1.assign(4, 2);

    EXPECT_EQ(2, v1.data()[0]);
    EXPECT_EQ(2, v1.data()[1]);

    EXPECT_EQ(2, v1.eigen()[0]);
    EXPECT_EQ(2, v1.eigen()[1]);
}

TEST(EigenArraysTest, TestVector1dResize)
{
    vectord v1 = { 6, 7, 8, 9 };
    v1.resize(2);

    EXPECT_EQ(2, v1.size());
    EXPECT_EQ(2, v1.eigen().size());
}

TEST(EigenArraysTest, TestVector1dAdvancedDataStructure) {
    //Eigen::matrix<std::map<int, double>, 3, 1> vectmap;
    vector<std::map<int, double>> vectmap;

    std::map<int, double> map1;
    map1.insert(std::map<int, double>::value_type(0, 10));
    map1.insert(std::map<int, double>::value_type(1, 20));
    map1.insert(std::map<int, double>::value_type(2, 30));
    vectmap.push_back(map1);
    //vectmap[0] = map1;

    std::map<int, double> map2;
    map2.insert(std::map<int, double>::value_type(0, 11));
    map2.insert(std::map<int, double>::value_type(1, 21));
    map2.insert(std::map<int, double>::value_type(2, 31));
    vectmap.push_back(map2);
    //vectmap[1] = map2;

    std::map<int, double> map3;
    map3.insert(std::map<int, double>::value_type(0, 12));
    map3.insert(std::map<int, double>::value_type(1, 22));
    map3.insert(std::map<int, double>::value_type(2, 32));
    vectmap.push_back(map3);
    //vectmap[2] = map3;

    EXPECT_EQ(10, vectmap[0][0]);
    EXPECT_EQ(20, vectmap[0][1]);
    EXPECT_EQ(30, vectmap[0][2]);

    EXPECT_EQ(11, vectmap[1][0]);
    EXPECT_EQ(21, vectmap[1][1]);
    EXPECT_EQ(31, vectmap[1][2]);

    EXPECT_EQ(12, vectmap[2][0]);
    EXPECT_EQ(22, vectmap.data()[2][1]);
    EXPECT_EQ(32, vectmap.eigen()[2][2]);
}

TEST(EigenArraysTest, TestVector1dEraseElement) {
    vectord dvec = { 1, 2, 3, 4, 5, 6 };

    dvec.erase(dvec.begin()); // erase first element
    EXPECT_EQ(5, dvec.size());
    EXPECT_EQ(2, dvec[0]);
    EXPECT_EQ(3, dvec[1]);
    EXPECT_EQ(4, dvec[2]);
    EXPECT_EQ(5, dvec[3]);
    EXPECT_EQ(6, dvec[4]);
    EXPECT_EQ(2, dvec.eigen()[0]);
    EXPECT_EQ(3, dvec.eigen()[1]);
    EXPECT_EQ(4, dvec.eigen()[2]);
    EXPECT_EQ(5, dvec.eigen()[3]);
    EXPECT_EQ(6, dvec.eigen()[4]);

    dvec.erase(dvec.end() - 1); // erase last element
    EXPECT_EQ(4, dvec.size());
    EXPECT_EQ(2, dvec[0]);
    EXPECT_EQ(3, dvec[1]);
    EXPECT_EQ(4, dvec[2]);
    EXPECT_EQ(5, dvec[3]);

    dvec.erase(dvec.begin() + 1); // erase second element
    EXPECT_EQ(3, dvec.size());
    EXPECT_EQ(2, dvec[0]);
    EXPECT_EQ(4, dvec[1]);
    EXPECT_EQ(5, dvec[2]);
}

TEST(EigenArraysTest, TestVector1dEraseElements)
{
    vectord dvec = { 1, 2, 3, 4, 5, 6 };

    dvec.erase(dvec.begin(), dvec.begin() + 3); // erase first three elements
    EXPECT_EQ(3, dvec.size());
    EXPECT_EQ(4, dvec[0]);
    EXPECT_EQ(5, dvec[1]);
    EXPECT_EQ(6, dvec[2]);

    dvec.erase(dvec.end() - 2, dvec.end()); // erase last two elements
    EXPECT_EQ(1, dvec.size());
    EXPECT_EQ(4, dvec[0]);
}

TEST(EigenArraysTest, TestVector1dEqualAssignment) {
    vectord v1 = { 1, 2, 3 };
    vectord v2;

    v2 = v1;

    EXPECT_EQ(3, v2.size());
    EXPECT_EQ(3, v2.eigen().size());
    EXPECT_EQ(1, v2[0]);
    EXPECT_EQ(2, v2[1]);
    EXPECT_EQ(3, v2[2]);
    EXPECT_EQ(1, v2.data()[0]);
    EXPECT_EQ(2, v2.data()[1]);
    EXPECT_EQ(3, v2.data()[2]);
    EXPECT_EQ(1, v2.eigen()[0]);
    EXPECT_EQ(2, v2.eigen()[1]);
    EXPECT_EQ(3, v2.eigen()[2]);
}

//#####################################

TEST(EigenArraysTest, TestVector2dConstructors) {
    // ctor with size
    matrixd m1(2, 3);
    EXPECT_EQ(2, m1.rows());
    EXPECT_EQ(3, m1.cols());
    EXPECT_EQ(2, m1.eigen().rows());
    EXPECT_EQ(3, m1.eigen().cols());
    EXPECT_EQ(0, m1(0, 0));
    EXPECT_EQ(0, m1(0, 1));
    EXPECT_EQ(0, m1(0, 2));
    EXPECT_EQ(0, m1(1, 0));
    EXPECT_EQ(0, m1(1, 1));
    EXPECT_EQ(0, m1(1, 2));
    for (const auto& e : m1)
        EXPECT_EQ(0, e);

    // ctor with default values from vector1d
    vectord v1 = {1, 2, 3};
    matrixd m11(2, v1);
    EXPECT_EQ(2, m11.rows());
    EXPECT_EQ(3, m11.cols());
    EXPECT_EQ(2, m11.eigen().rows());
    EXPECT_EQ(3, m11.eigen().cols());
    EXPECT_EQ(1, m11(0, 0));
    EXPECT_EQ(2, m11(0, 1));
    EXPECT_EQ(3, m11(0, 2));
    EXPECT_EQ(1, m11(1, 0));
    EXPECT_EQ(2, m11(1, 1));
    EXPECT_EQ(3, m11(1, 2));

    // ctor with default values from std::vector
    std::vector<double> v2 = { 1, 2, 3 };
    matrixd m12(2, v2);
    EXPECT_EQ(2, m12.rows());
    EXPECT_EQ(3, m12.cols());
    EXPECT_EQ(2, m12.eigen().rows());
    EXPECT_EQ(3, m12.eigen().cols());
    EXPECT_EQ(1, m12(0, 0));
    EXPECT_EQ(2, m12(0, 1));
    EXPECT_EQ(3, m12(0, 2));
    EXPECT_EQ(1, m12(1, 0));
    EXPECT_EQ(2, m12(1, 1));
    EXPECT_EQ(3, m12(1, 2));

    // ctor with default values
    matrixd m2(2, 3, 1);
    EXPECT_EQ(2, m2.rows());
    EXPECT_EQ(3, m2.cols());
    EXPECT_EQ(2, m2.eigen().rows());
    EXPECT_EQ(3, m2.eigen().cols());
    EXPECT_EQ(1, m2(0, 0));
    EXPECT_EQ(1, m2(0, 1));
    EXPECT_EQ(1, m2(0, 2));
    EXPECT_EQ(1, m2(1, 0));
    EXPECT_EQ(1, m2(1, 1));
    EXPECT_EQ(1, m2(1, 2));
    for (const auto& e : m2)
        EXPECT_EQ(1, e);

    // empty ctor
    matrixd m3;
    EXPECT_EQ(0, m3.rows());
    EXPECT_EQ(0, m3.cols());
    EXPECT_EQ(0, m3.eigen().rows());
    EXPECT_EQ(0, m3.eigen().cols());

    // copy ctor
    matrixd m4(m2);
    EXPECT_EQ(2, m4.rows());
    EXPECT_EQ(3, m4.cols());
    EXPECT_EQ(2, m4.eigen().rows());
    EXPECT_EQ(3, m4.eigen().cols());
    EXPECT_EQ(1, (m4.data())[0]);

    // initializer list
    matrixd m5 = { {1,2,3}, {4,5,6} };
    EXPECT_EQ(2, m5.rows());
    EXPECT_EQ(3, m5.cols());
    EXPECT_EQ(2, m5.eigen().rows());
    EXPECT_EQ(3, m5.eigen().cols());
    EXPECT_EQ(1, m5(0, 0));
    EXPECT_EQ(2, m5(0, 1));
    EXPECT_EQ(3, m5(0, 2));
    EXPECT_EQ(4, m5(1, 0));
    EXPECT_EQ(5, m5(1, 1));
    EXPECT_EQ(6, m5(1, 2));
    EXPECT_EQ(1, m5.data()[0]);
    EXPECT_EQ(2, m5.data()[1]);
    EXPECT_EQ(3, m5.data()[2]);
    EXPECT_EQ(4, m5.data()[3]);
    EXPECT_EQ(5, m5.data()[4]);
    EXPECT_EQ(6, m5.data()[5]);

    vectord row1 = {1, 2, 3};
    vectord row2 = {4, 5, 6};
    matrixd m6 = { row1, row2};
    EXPECT_EQ(2, m6.rows());
    EXPECT_EQ(3, m6.cols());
    EXPECT_EQ(2, m6.eigen().rows());
    EXPECT_EQ(3, m6.eigen().cols());
    EXPECT_EQ(1, m6(0, 0));
    EXPECT_EQ(2, m6(0, 1));
    EXPECT_EQ(3, m6(0, 2));
    EXPECT_EQ(4, m6(1, 0));
    EXPECT_EQ(5, m6(1, 1));
    EXPECT_EQ(6, m6(1, 2));
    EXPECT_EQ(1, m6.data()[0]);
    EXPECT_EQ(2, m6.data()[1]);
    EXPECT_EQ(3, m6.data()[2]);
    EXPECT_EQ(4, m6.data()[3]);
    EXPECT_EQ(5, m6.data()[4]);
    EXPECT_EQ(6, m6.data()[5]);

    // from eigen matrix
    matrixd m7(m6.eigen());
    EXPECT_EQ(2, m7.rows());
    EXPECT_EQ(3, m7.cols());
    EXPECT_EQ(2, m7.eigen().rows());
    EXPECT_EQ(3, m7.eigen().cols());
    EXPECT_EQ(1, m7(0, 0));
    EXPECT_EQ(2, m7(0, 1));
    EXPECT_EQ(3, m7(0, 2));
    EXPECT_EQ(4, m7(1, 0));
    EXPECT_EQ(5, m7(1, 1));
    EXPECT_EQ(6, m7(1, 2));
    EXPECT_EQ(1, m7.data()[0]);
    EXPECT_EQ(2, m7.data()[1]);
    EXPECT_EQ(3, m7.data()[2]);
    EXPECT_EQ(4, m7.data()[3]);
    EXPECT_EQ(5, m7.data()[4]);
    EXPECT_EQ(6, m7.data()[5]);
}

TEST(EigenArraysTest, TestVector2dIterators) {
    matrixd m1 = { {1,2,3}, {4,5,6} };

    int i = 0;
    for (const auto& e : m1)
        EXPECT_EQ(++i, e);


    matrixd m2(2, 3);
    i = 0;
    for (auto& e : m2)
        e = ++i;

    i = 0;
    for (const auto& e : m2)
        EXPECT_EQ(++i, e);
}

TEST(EigenArraysTest, TestVector2dEigenMap) {
    vectord row1 = { 1, 2, 3 };
    vectord row2 = { 4, 5, 6 };
    matrixd m6 = { row1, row2 };

    // manipulate the eigen data structure
    // since the eigen data is mapped to the internal data structure (std::vector) of vector2d
    // the entries of the internal data structure should change too.
    m6.eigen()(0, 0) = 7;
    m6.eigen()(0, 1) = 8;
    m6.eigen()(0, 2) = 9;
    m6.eigen()(1, 0) = 10;
    m6.eigen()(1, 1) = 11;
    m6.eigen()(1, 2) = 12;

    EXPECT_EQ(2, m6.rows());
    EXPECT_EQ(3, m6.cols());
    EXPECT_EQ(2, m6.eigen().rows());
    EXPECT_EQ(3, m6.eigen().cols());
    EXPECT_EQ(7, m6(0, 0));
    EXPECT_EQ(8, m6(0, 1));
    EXPECT_EQ(9, m6(0, 2));
    EXPECT_EQ(10, m6(1, 0));
    EXPECT_EQ(11, m6(1, 1));
    EXPECT_EQ(12, m6(1, 2));
    EXPECT_EQ(7, m6.data()[0]);
    EXPECT_EQ(8, m6.data()[1]);
    EXPECT_EQ(9, m6.data()[2]);
    EXPECT_EQ(10, m6.data()[3]);
    EXPECT_EQ(11, m6.data()[4]);
    EXPECT_EQ(12, m6.data()[5]);
}

TEST(EigenArraysTest, TestVector2dPushBack) {
    matrixd m1 = { {1,2,3}, {4,5,6} };
    vectord v1 = { 7,8,9 };
    std::vector<double> v2 = { 10,11,12 };
    m1.push_back(v1);
    m1.push_back(v2);

    EXPECT_EQ(4, m1.rows());
    EXPECT_EQ(3, m1.cols());
    EXPECT_EQ(4, m1.eigen().rows());
    EXPECT_EQ(3, m1.eigen().cols());
    EXPECT_EQ(1, m1(0, 0));
    EXPECT_EQ(2, m1(0, 1));
    EXPECT_EQ(3, m1(0, 2));
    EXPECT_EQ(4, m1(1, 0));
    EXPECT_EQ(5, m1(1, 1));
    EXPECT_EQ(6, m1(1, 2));
    EXPECT_EQ(7, m1(2, 0));
    EXPECT_EQ(8, m1(2, 1));
    EXPECT_EQ(9, m1(2, 2));
    EXPECT_EQ(10, m1(3, 0));
    EXPECT_EQ(11, m1(3, 1));
    EXPECT_EQ(12, m1(3, 2));

    EXPECT_EQ(1, m1.data()[0]);
    EXPECT_EQ(2, m1.data()[1]);
    EXPECT_EQ(3, m1.data()[2]);
    EXPECT_EQ(4, m1.data()[3]);
    EXPECT_EQ(5, m1.data()[4]);
    EXPECT_EQ(6, m1.data()[5]);
    EXPECT_EQ(7, m1.data()[6]);
    EXPECT_EQ(8, m1.data()[7]);
    EXPECT_EQ(9, m1.data()[8]);
    EXPECT_EQ(10, m1.data()[9]);
    EXPECT_EQ(11, m1.data()[10]);
    EXPECT_EQ(12, m1.data()[11]);
}

TEST(EigenArraysTest, TestVector2dClear) {
    matrixd m1 = { {1,2,3}, {4,5,6} };
    
    m1.clear();
    EXPECT_EQ(0, m1.rows());
    EXPECT_EQ(0, m1.cols());
    EXPECT_EQ(0, m1.eigen().rows());
    EXPECT_EQ(0, m1.eigen().cols());
}

TEST(EigenArraysTest, TestVector2dReset) {
    matrixd m1 = { {1,2,3}, {4,5,6} };

    m1.reset();
    EXPECT_EQ(2, m1.rows());
    EXPECT_EQ(3, m1.cols());
    EXPECT_EQ(2, m1.eigen().rows());
    EXPECT_EQ(3, m1.eigen().cols());

    for (const auto& e : m1)
        EXPECT_EQ(0, e);
}

TEST(EigenArraysTest, TestVector2dAssign) {
    matrixd m1 = { {1,2,3}, {4,5,6} };

    m1.assign(3, 4, -10);
    EXPECT_EQ(3, m1.rows());
    EXPECT_EQ(4, m1.cols());
    EXPECT_EQ(3, m1.eigen().rows());
    EXPECT_EQ(4, m1.eigen().cols());

    for (const auto& e : m1)
        EXPECT_EQ(-10, e);
}

TEST(EigenArraysTest, TestVector2dResize) {
    matrixd m1 = { {1,2,3}, {4,5,6} };

    m1.resize(3, 4);
    EXPECT_EQ(3, m1.rows());
    EXPECT_EQ(4, m1.cols());
    EXPECT_EQ(3, m1.eigen().rows());
    EXPECT_EQ(4, m1.eigen().cols());

    EXPECT_EQ(1, m1(0, 0));
    EXPECT_EQ(2, m1(0, 1));
    EXPECT_EQ(3, m1(0, 2));
    EXPECT_EQ(4, m1(0, 3));

    EXPECT_EQ(5, m1(1, 0));
    EXPECT_EQ(6, m1(1, 1));
    EXPECT_EQ(0, m1(1, 2));
    EXPECT_EQ(0, m1(1, 3));

    EXPECT_EQ(0, m1(2, 0));
    EXPECT_EQ(0, m1(2, 1));
    EXPECT_EQ(0, m1(2, 2));
    EXPECT_EQ(0, m1(2, 3));
}

TEST(EigenArraysTest, TestVector2dReserveAndAt) {
    matrixd m0 = { {1,2,3}, {4,5,6} }, m1;
    
    // reserve does not change size!
    m1.reserve(m0.rows(), m0.cols());
    for (size_t r = 0; r < m0.rows(); ++r) {
        vectord vec(m0.cols());
        for (size_t c = 0; c < m0.cols(); ++c)
            vec.at(c) = m0(r, c);

        // however a push_back does.
        m1.push_back(vec);
    }

    EXPECT_EQ(2, m1.rows());
    EXPECT_EQ(3, m1.cols());
    EXPECT_EQ(2, m1.eigen().rows());
    EXPECT_EQ(3, m1.eigen().cols());
    EXPECT_EQ(1, m1.at(0, 0));
    EXPECT_EQ(2, m1.at(0, 1));
    EXPECT_EQ(3, m1.at(0, 2));
    EXPECT_EQ(4, m1.at(1, 0));
    EXPECT_EQ(5, m1.at(1, 1));
    EXPECT_EQ(6, m1.at(1, 2));
}

TEST(EigenArraysTest, TestVector2dEqualOperator) {
    matrixd m0 = { {1,2,3}, {4,5,6} }, m1, m2;

    m1 = m0;

    EXPECT_EQ(2, m1.rows());
    EXPECT_EQ(3, m1.cols());
    EXPECT_EQ(2, m1.eigen().rows());
    EXPECT_EQ(3, m1.eigen().cols());
    EXPECT_EQ(1, m1.at(0, 0));
    EXPECT_EQ(2, m1.at(0, 1));
    EXPECT_EQ(3, m1.at(0, 2));
    EXPECT_EQ(4, m1.at(1, 0));
    EXPECT_EQ(5, m1.at(1, 1));
    EXPECT_EQ(6, m1.at(1, 2));

    m2 = std::move(m0);

    EXPECT_EQ(2, m2.rows());
    EXPECT_EQ(3, m2.cols());
    EXPECT_EQ(2, m2.eigen().rows());
    EXPECT_EQ(3, m2.eigen().cols());
    EXPECT_EQ(1, m2.at(0, 0));
    EXPECT_EQ(2, m2.at(0, 1));
    EXPECT_EQ(3, m2.at(0, 2));
    EXPECT_EQ(4, m2.at(1, 0));
    EXPECT_EQ(5, m2.at(1, 1));
    EXPECT_EQ(6, m2.at(1, 2));
}

TEST(EigenArraysTest, TestVector2dAdvancedDataStructure) {
    matrix<std::string> m0 = { {std::string("Das ist ein sehr langer String, der in einer Matrix gespeichert wurde."), 
                                    std::string("Das ist ein sehr langer String, der in einer Matrix gespeichert wurde. (2)")},
                                {std::string("Das ist ein weiterer langer String, der in einer Matrix gespeichert wurde."),
                                    std::string("Das ist ein weiterer langer String, der in einer Matrix gespeichert wurde. (2)")} };

    for (const auto& e : m0)
        std::cout << e << std::endl;

    EXPECT_EQ(2, m0.rows());
    EXPECT_EQ(2, m0.cols());
    EXPECT_EQ(2, m0.eigen().rows());
    EXPECT_EQ(2, m0.eigen().cols());
    EXPECT_EQ(70, m0(0, 0).size());
    EXPECT_EQ(74, m0(0, 1).size());
    EXPECT_EQ(74, m0(1, 0).size());
    EXPECT_EQ(78, m0(1, 1).size());

    vectord v1 = { 1, 2 };
    vectord v2 = { 3, 4 };
    vectord v3 = { 5, 6 };
    vectord v4 = { 7, 8 };
    matrix<vector<double>> tensor = { {v1, v2}, {v3, v4} };

    EXPECT_EQ(2, tensor.rows());
    EXPECT_EQ(2, tensor.cols());
    EXPECT_EQ(2, tensor.eigen().rows());
    EXPECT_EQ(2, tensor.eigen().cols());
    EXPECT_EQ(1, tensor(0, 0)[0]);
    EXPECT_EQ(2, tensor(0, 0)[1]);
    EXPECT_EQ(3, tensor(0, 1)[0]);
    EXPECT_EQ(4, tensor(0, 1)[1]);
    EXPECT_EQ(5, tensor(1, 0)[0]);
    EXPECT_EQ(6, tensor(1, 0)[1]);
    EXPECT_EQ(7, tensor(1, 1)[0]);
    EXPECT_EQ(8, tensor(1, 1)[1]);
}

//#####################################

TEST(EigenArraysTest, TestVector2dmatrixMultiplicationTime) {
    srand((int)time(NULL));

    const size_t nsize = 128;
    const size_t nexec = 100;

    std::vector<std::vector<double>> cmat1(nsize, std::vector<double>(nsize)), cmat2(nsize, std::vector<double>(nsize));
    for (size_t c = 0; c < nsize; ++c)
        for (size_t r = 0; r < nsize; ++r)
        {
            cmat1[r][c] = rand() % 10; // random values
            cmat2[r][c] = rand() % 10; // to prevent compiler optimization
        }
    auto t_start = std::chrono::high_resolution_clock::now();
    for (size_t e = 0; e < nexec; ++e)
    {
        std::vector<std::vector<double>> cmat(nsize, std::vector<double>(nsize));
        for (size_t c = 0; c < nsize; ++c)
            for (size_t r = 0; r < nsize; ++r)
                for (size_t i = 0; i < nsize; ++i)
                    cmat[r][c] += cmat1[r][c] * cmat2[r][c];
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "std::vector mulitplication took " << t1 / nexec << "ms" << std::endl;

    Eigen::MatrixXd emat1(nsize, nsize), emat2(nsize, nsize);
    for (size_t c = 0; c < nsize; ++c)
        for (size_t r = 0; r < nsize; ++r)
        {
            emat1(r, c) = rand() % 10;
            emat2(r, c) = rand() % 10;
        }
    t_start = std::chrono::high_resolution_clock::now();
    for (size_t e = 0; e < nexec; ++e)
    {
        Eigen::MatrixXd emat(nsize, nsize);
        emat = emat1 * emat2;
    }
    t_end = std::chrono::high_resolution_clock::now();
    double t2 = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Eigen mulitplication took " << t2 / nexec << "ms" << std::endl;

    matrix<double> dmat1(nsize, nsize), dmat2(nsize, nsize);
    for (size_t c = 0; c < nsize; ++c)
        for (size_t r = 0; r < nsize; ++r)
        {
            dmat1(r, c) = rand() % 10;
            dmat2(r, c) = rand() % 10;
        }
    t_start = std::chrono::high_resolution_clock::now();
    for (size_t e = 0; e < nexec; ++e)
    {
        matrix<double> dmat(nsize, nsize);
        dmat = dmat1 * dmat2;
    }
    t_end = std::chrono::high_resolution_clock::now();
    double t3 = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "vector2d mulitplication took " << t3 / nexec << "ms" << std::endl;

    // multiplication with Eigen should be faster
    EXPECT_LT(t3, t1);
    EXPECT_LT(t2, t1);
}

TEST(EigenArraysTest, TestmatrixMultiplication) {
    vectord drow1 = { 1, 2};
    vectord drow2 = { 3, 4};
    matrix<double> dmat1 = { drow1, drow2 };

    vectord drow3 = { 5, 6};
    vectord drow4 = { 7, 8};
    matrix<double> dmat2 = { drow3, drow4 };

    matrix<double> dmat = dmat1 * dmat2;
    EXPECT_EQ(19, dmat(0, 0));
    EXPECT_EQ(22, dmat(0, 1));
    EXPECT_EQ(43, dmat(1, 0));
    EXPECT_EQ(50, dmat(1, 1));

    vectord dvec = dmat1 * drow3;
    EXPECT_EQ(17, dvec[0]);
    EXPECT_EQ(39, dvec[1]);

    dmat = dmat1 * 2.0;
    dmat = 3.0 * dmat;
    EXPECT_EQ(6, dmat(0, 0));
    EXPECT_EQ(12, dmat(0, 1));
    EXPECT_EQ(18, dmat(1, 0));
    EXPECT_EQ(24, dmat(1, 1));
}

TEST(EigenArraysTest, TestmatrixTranspose) {
    matrix<double> dmat(2, 2); 
    dmat.eigen() << 1, 2,
                    3, 4;

    EXPECT_EQ(1, dmat(0, 0));
    EXPECT_EQ(2, dmat(0, 1));
    EXPECT_EQ(3, dmat(1, 0));
    EXPECT_EQ(4, dmat(1, 1));

    dmat = eigen::transpose(dmat);

    EXPECT_EQ(1, dmat(0, 0));
    EXPECT_EQ(2, dmat(1, 0));
    EXPECT_EQ(3, dmat(0, 1));
    EXPECT_EQ(4, dmat(1, 1));
}

TEST(EigenArraysTest, TestvectorMultiplication) {
    vectord v1 = {1, 2, 3};
    
    vectord v2 = 2.0 * v1;
    v2 = v2 * 3.0;

    EXPECT_EQ(6, v2[0]);
    EXPECT_EQ(12, v2[1]);
    EXPECT_EQ(18, v2[2]);

    vectord v3 = { 4, 5, 6 };
    double d1 = v1 * v3;

    EXPECT_EQ(32, d1);
}

TEST(EigenArraysTest, TestmatrixAdditionSubtraction) {
    vectord drow1 = { 1, 2, 3 };
    vectord drow2 = { 4, 5, 6 };
    matrixd dmat1 = { drow1, drow2 };

    vectord drow3 = { -1, -2, -3 };
    vectord drow4 = { -4, -5, -6 };
    matrix<double> dmat2 = { drow3, drow4 };

    matrix<double> dmat = dmat1 + dmat2;

    for (const auto& e : dmat)
        EXPECT_EQ(0, e);

    dmat = dmat + dmat1;
    size_t s = 0;
    for (const auto& e : dmat)
        EXPECT_EQ(++s, e);


    matrix<double> dmat3 = dmat1 - dmat1;

    for (const auto& e : dmat3)
        EXPECT_EQ(0, e);

    dmat3 = dmat3 - dmat2;
    s = 0;
    for (const auto& e : dmat3)
        EXPECT_EQ(++s, e);
}

TEST(EigenArraysTest, TestvectorAdditionSubtraction) {
    vectord v1 = { 1, 2, 3 };
    vectord v2 = { 4, 5, 6 };
    vectord v = v1 + v2;

    EXPECT_EQ(5, v[0]);
    EXPECT_EQ(7, v[1]);
    EXPECT_EQ(9, v[2]);

    v = v1 - v2;

    EXPECT_EQ(-3, v[0]);
    EXPECT_EQ(-3, v[1]);
    EXPECT_EQ(-3, v[2]);
}

TEST(EigenArraysTest, CoefficientWiseMultiplication) {
    vectord v1 = { 1, 2, 3 };
    vectord v2 = { 4, 5, 6 };
    vectord v3 = eigen::cprod(v1, v2);

    EXPECT_EQ(4, v3[0]);
    EXPECT_EQ(10, v3[1]);
    EXPECT_EQ(18, v3[2]);
}

TEST(EigenArraysTest, PlusEq) {
    vectord v1 = { 1, 2, 3 };
    vectord v2 = { 4, 5, 6 };
    
    v1 += v2;
    v2 -= v1;

    EXPECT_EQ(5, v1[0]);
    EXPECT_EQ(7, v1[1]);
    EXPECT_EQ(9, v1[2]);

    EXPECT_EQ(-1, v2[0]);
    EXPECT_EQ(-2, v2[1]);
    EXPECT_EQ(-3, v2[2]);
}

TEST(EigenArraysTest, NormTest) {
    vectord vec(11);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = i;

    eigen::normalize(vec);
    double n = eigen::norm(vec);
    std::cout << "norm = " << n << std::endl;
    EXPECT_NEAR(1, n, 0.001);

    eigen::normalize<1>(vec);
    double n1 = eigen::norm<1>(vec);
    std::cout << "normL1 = " << n1 << std::endl;
    EXPECT_NEAR(1, n1, 0.001);

    eigen::normalize<2>(vec);
    double n2 = eigen::norm<2>(vec);
    std::cout << "normL2 = " << n2 << std::endl;
    EXPECT_NEAR(1, n2, 0.001);

    eigen::normalize<3>(vec);
    double n3 = eigen::norm<3>(vec);
    std::cout << "normL3 = " << n3 << std::endl;
    EXPECT_NEAR(1, n3, 0.001);
}

TEST(EigenArrayTest, AuxMultiplication) {
    math::vector<double> a = {1, 2, 3};
    math::matrix<double> M = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    auto b = M * a;
    
    EXPECT_EQ(2, b[0]);
    EXPECT_EQ(1, b[1]);
    EXPECT_EQ(3, b[2]);
}

TEST(EigenArrayTest, MemoryMappingVectors) {
    struct nn {
        math::vector<double> parameters = {1, 2, 3, 4, 5, 6};
        math::vector<double> iweights = math::vector<double>(3);
        math::vector<double> oweights = math::vector<double>(3);

        // map the memory such that: memory(parameters) = memory(iweights) + memory(oweights)
        void map() {
            // this does not call the copy constructor. In fact only the memory locations are mapped.
            new (&iweights.eigen()) math::vector<double>::map_type(parameters.data(), 3, 1);
            new (&oweights.eigen()) math::vector<double>::map_type(parameters.data() + 3, 3, 1);
        }
    } nn;

    // init
    nn.map();

    // the vectors iweights and oweights now should contain the values of "parameters"
    EXPECT_EQ(1, nn.iweights.eigen()[0]);
    EXPECT_EQ(2, nn.iweights[1]);
    EXPECT_EQ(3, nn.iweights[2]);
    EXPECT_EQ(4, nn.oweights[0]);
    EXPECT_EQ(5, nn.oweights[1]);
    EXPECT_EQ(6, nn.oweights[2]);


    std::cout << "Modified" << std::endl;
    math::matrix<double> M = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    nn.iweights = M * nn.iweights;
    nn.parameters[5] = -7;

    std::cout << nn.parameters << std::endl << std::endl;
    std::cout << nn.iweights << std::endl;
    std::cout << nn.oweights << std::endl;
}