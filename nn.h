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
    typedef struct ilayer {
        double input;            // mat[NINPUTS]
        double weight;           // mat[NINPUTS]
        vector<double> output;   // mat[NINPUTS][NNEURONS]
        double theta;            // mat[NINPUTS]
    } ilayer;

    typedef struct neuron {
        vector<double> input;   // mat[NNEURONS][NINPUTS]
        vector<double> weight;  // mat[NNEURONS][NINPUTS]
        vector<double> output;  // mat[NNEURONS][NOUTPUTS]
        double theta;           // mat[NNEURONS]
    } neuron;

    typedef struct olayer {
        vector<double> input;    // mat[NOUTPUTS][NNEURONS]
        vector<double> weight;   // mat[NOUTPUTS][NNEURONS]
        double output;           // mat[NOUTPUTS]
        double theta;            // mat[NOUTPUTS]
    } olayer;

    typedef struct nn {
        nn(size_t _ninputs, size_t _noutputs, size_t _nneurons)
            : parameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs),
            iweights(parameters.data(), _ninputs, 1), 
            itheta(parameters.data() + _ninputs, _ninputs, 1),
            nweights(parameters.data() + 2 * _ninputs, _nneurons, _ninputs),
            ntheta(parameters.data() +  2 * _ninputs + _nneurons * _ninputs, _nneurons, 1),
            oweights(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons, _noutputs, _nneurons),
            otheta(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons + _noutputs * _nneurons, _noutputs, 1),
            ninputs(_ninputs), noutputs(_noutputs), nneurons(_nneurons) {}

        // all parameters of the network
        vector<double> parameters;
        // parameters of the input neurons
        vector<double>::map_type iweights; // mat[NINPUTS] -> par(0, ninputs)
        vector<double>::map_type itheta;   // mat[NINPUTS] -> par(ninputs, ninputs + ninputs)
        // parameters of the fully connected, inner neurons
        matrix<double>::vector_map_type nweights; // mat[NNEURONS][NINPUTS] -> par(2 * ninputs, 2 * ninputs + nneurons * ninputs)
        vector<double>::map_type ntheta;   // mat[NNEURONS] -> par(2 * ninputs + nneurons * ninputs, 2 * ninputs + nneurons * ninputs + nneurons)
        // parameters of the output neurons
        matrix<double>::vector_map_type oweights; // mat[NOUTPUTS][NNEURONS] -> par(2 * ninputs + nneurons * ninputs + nneurons, 2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons)
        vector<double>::map_type otheta;   // mat[NOUTPUTS] -> par(2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons, 2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons + noutputs)

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
                nn.iweights(1) = 3;
                std::cout << nn.iweights(0) << std::endl;
                std::cout << nn.itheta(0) << std::endl;
                std::cout << nn.parameters[1] << std::endl;
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

            static double rnd(double a, double b){
                double x = (double)rand() / (double)(RAND_MAX) * (b - a) + a;
                return x;
            }
            
    };
}