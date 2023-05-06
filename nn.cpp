/*
 *  nn.c
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#include "nn.h"
#include <cmath>

namespace math {
    void supervisor::init() {

    }

    double supervisor::transferFunction(double x, double theta) {
        #ifdef SIGMOID
            return (1 / (1 + std::exp(theta - x)));
        #endif

        #ifdef RELU
            if(x + theta >= 0)
                return x + theta;
            return  0;
        #endif

        #ifdef TANH
            return std::tanh(theta + x);
        #endif
    }
}
