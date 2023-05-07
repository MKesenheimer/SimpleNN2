/*
 *  main.cpp
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "nn.h"

#include "vector.h"

int main(int argc, char* args[]) {

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> mat1(nullptr, 10, 1);
    std::cout << mat1.cols() << std::endl;
    std::cout << mat1.rows() << std::endl << std::endl;

    Eigen::Matrix<double, Eigen::Dynamic, 1> mat2(10);
    std::cout << mat2.cols() << std::endl;
    std::cout << mat2.rows() << std::endl;
    std::cout << mat2[0] << std::endl << std::endl;


    std::vector<double> v3 = {1, 2, 3};
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> mat3(v3.data(), v3.size(), 1);
    std::cout << mat3.cols() << std::endl;
    std::cout << mat3.rows() << std::endl;
    std::cout << mat3[0] << std::endl;
    std::cout << mat3[1] << std::endl;
    std::cout << mat3[2] << std::endl << std::endl;

    math::vector<double> v4(v3);
    std::cout << v4[0] << std::endl;
    std::cout << v4[1] << std::endl;
    std::cout << v4[2] << std::endl << std::endl;

    return 0;
}