/*
 *  main.cpp
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "nn.h"

int main(int argc, char* args[]) {
    // initialize random numbers
    srand((unsigned int)time(NULL));

    const size_t ninputs = 4, noutputs = 3;
    math::nn nn(ninputs, noutputs, 40);

    math::supervisor::init(nn);

    // train
    std::vector<math::dataSet> dataset;
    math::dataSet d(ninputs, noutputs);
    d.xx[0] = 0;
    d.xx[1] = 0;
    d.xx[2] = 0;
    d.xx[3] = 0;
    d.yy[0] = 0;
    d.yy[1] = 0;
    d.yy[2] = 0;
    dataset.push_back(d);
    d.xx[0] = 0;
    d.xx[1] = 1;
    d.xx[2] = 0;
    d.xx[3] = 1;
    d.yy[0] = 0;
    d.yy[1] = 1;
    d.yy[2] = 0;
    dataset.push_back(d);
    d.xx[0] = 1;
    d.xx[1] = 0;
    d.xx[2] = 1;
    d.xx[3] = 0;
    d.yy[0] = 1;
    d.yy[1] = 0;
    d.yy[2] = 0;
    dataset.push_back(d);
    d.xx[0] = 1;
    d.xx[1] = 1;
    d.xx[2] = 1;
    d.xx[3] = 1;
    d.yy[0] = 1;
    d.yy[1] = 1;
    d.yy[2] = 0;
    dataset.push_back(d);
    /*d.xx[0] = 1;
    d.xx[1] = 1;
    d.xx[2] = 0;
    d.xx[3] = 0;
    d.yy[0] = 0;
    d.yy[1] = 0;
    d.yy[2] = 1;
    dataset.push_back(d);
    d.xx[0] = 0;
    d.xx[1] = 0;
    d.xx[2] = 1;
    d.xx[3] = 1;
    d.yy[0] = 1;
    d.yy[1] = 0;
    d.yy[2] = 1;
    dataset.push_back(d);*/
    math::supervisor::train(nn, dataset, 0.001, 15);

    // test
    math::vector<double> x1({0, 0, 0, 0});
    math::supervisor::calculateNN(x1, nn);
    std::cout << "o1 = " << nn.ooutput[0] << std::endl;
    std::cout << "o1 = " << nn.ooutput[1] << std::endl;
    std::cout << "o2 = " << nn.ooutput[2] << std::endl << std::endl;

    math::vector<double> x2({1, 1, 1, 1});
    math::supervisor::calculateNN(x2, nn);
    std::cout << "o1 = " << nn.ooutput[0] << std::endl;
    std::cout << "o1 = " << nn.ooutput[1] << std::endl;
    std::cout << "o2 = " << nn.ooutput[2] << std::endl;

    return 0;
}