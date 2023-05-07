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
    math::nn nn(4, 2, 4);

    math::supervisor::init(nn);
    return 0;
}