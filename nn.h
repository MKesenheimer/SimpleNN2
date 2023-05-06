/*
 *  nn.h
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#pragma once
#include "vector.h"
#include "matrix.h"
#include "operators.h"

// choose transfer function
//#define SIGMOID
#define RELU
//#define TANH

namespace math {
    typedef struct nn {
        vector<double> iweights;
        
    } nn;

    class supervisor {
        public:
            /// <summary>
            /// 
            /// </summary>
            void init();

        private:
            /// <summary>
            /// 
            /// </summary>
            double transferFunction(double x, double theta);
    };
}