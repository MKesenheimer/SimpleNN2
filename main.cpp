/*
 *  main.cpp
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#include <iostream>

#include "nn.h"
#include "vector.h"
#include "operators.h"

int main( int argc, char* args[]) {
    math::vector<double> a = {1, 2, 3};
    math::matrix<double> M = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    auto b = M * a;
    std::cout << b << std::endl;


    struct nn {
        math::vector<double> parameters = {1, 2, 3, 4, 5, 6};
        math::vector<double> iweights = math::vector<double>(3);
        math::vector<double> oweights = math::vector<double>(3);

        void map() {
            new (&iweights.eigen()) math::vector<double>::map_type(parameters.data(), 3, 1);
            new (&oweights.eigen()) math::vector<double>::map_type(parameters.data() + 3, 3, 1);
        }
    } nn;

    // init
    nn.map();

    //std::cout << nn.parameters << std::endl;
    std::cout << nn.iweights << std::endl;
    std::cout << nn.oweights << std::endl;

    std::cout << "Modified" << std::endl;
    //nn.iweights.eigen() = M.eigen() * nn.iweights.eigen();
    nn.iweights = M * nn.iweights;
    nn.parameters[5] = -7;

    std::cout << nn.parameters << std::endl << std::endl;
    std::cout << nn.iweights << std::endl;
    std::cout << nn.oweights << std::endl;

    return 0;
}