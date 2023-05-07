/*
 *  nn.h
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#pragma once
#include <cmath>

#include "vector.h"
#include "matrix.h"
#include "operators.h"

// choose transfer function
//#define SIGMOID
#define RELU
//#define TANH

namespace math {
    typedef struct nn {
        nn(size_t _ninputs, size_t _noutputs, size_t _nneurons)
            : parameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs), 
            iweights(parameters.data(), _ninputs, 1), 
            itheta(parameters.data() + _ninputs, _ninputs, 1), 
            ninputs(_ninputs), noutputs(_noutputs), nneurons(_nneurons) {}

        vector<double> parameters;
        vector<double>::map_type iweights;
        vector<double>::map_type itheta;

        const size_t ninputs, noutputs, nneurons;
    } nn;

    class supervisor {
        public:
            /// <summary>
            /// 
            /// </summary>
            static void init(nn& nn) {
                nn.parameters[0] = 1;
                nn.parameters[nn.ninputs] = 2;

                std::cout << nn.iweights(0) << std::endl;
                std::cout << nn.itheta(0) << std::endl;
            }

        private:
            /// <summary>
            /// 
            /// </summary>
            static double transferFunction(double x, double theta) {
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
            
    };
}