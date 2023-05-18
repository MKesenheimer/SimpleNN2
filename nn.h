/*
 *  nn.h
 *  Created by Matthias Kesenheimer on 05.05.23.
 *  Copyright 2023. All rights reserved.
 */

#pragma once
#include <cmath>
#include <vector>

#include "vector.h"
#include "matrix.h"
#include "operators.h"

// choose transfer function
//#define SIGMOID
//#define RELU
//#define TANH

namespace math {
    /// <summary>
    /// 
    /// </summary>
    typedef struct nn {
        /// <summary>
        /// 
        /// </summary>
        nn(size_t _ninputs, size_t _noutputs, size_t _nneurons)
            : parameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs),
            
            iweights(parameters.data(), _ninputs, 1), 
            itheta(parameters.data() + _ninputs, _ninputs, 1),
            ioutput(_ninputs, 0), // ioutput(_ninputs, _nneurons, 0),

            hweights(parameters.data() + 2 * _ninputs, _nneurons, _ninputs),
            htheta(parameters.data() +  2 * _ninputs + _nneurons * _ninputs, _nneurons, 1),
            houtput(_nneurons, 0), // houtput(_nneurons, _noutputs, 0),

            oweights(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons, _noutputs, _nneurons),
            otheta(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons + _noutputs * _nneurons, _noutputs, 1),
            ooutput(_noutputs, 0),

            ntotparameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs),
            ninputs(_ninputs), noutputs(_noutputs), nneurons(_nneurons) {}

        /// <summary>
        /// copy constructor
        /// </summary>
        nn(const vector<double>& _parameters, size_t _ninputs, size_t _noutputs, size_t _nneurons)
            : parameters(_parameters),

            iweights(parameters.data(), _ninputs, 1), 
            itheta(parameters.data() + _ninputs, _ninputs, 1),
            ioutput(_ninputs, 0),

            hweights(parameters.data() + 2 * _ninputs, _nneurons, _ninputs),
            htheta(parameters.data() +  2 * _ninputs + _nneurons * _ninputs, _nneurons, 1),
            houtput(_nneurons, 0),

            oweights(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons, _noutputs, _nneurons),
            otheta(parameters.data() + 2 * _ninputs + _nneurons * _ninputs + _nneurons + _noutputs * _nneurons, _noutputs, 1),
            ooutput(_noutputs, 0),

            ntotparameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs),
            ninputs(_ninputs), noutputs(_noutputs), nneurons(_nneurons) {}
        

        /// <summary>
        /// all parameters of the network
        /// </summary>
        vector<double> parameters;
        
        /// <summary>
        /// parameters of the input neurons
        /// </summary>
        vector<double>::map_type iweights; // mat[NINPUTS] -> par(0, ninputs)
        vector<double>::map_type itheta;   // mat[NINPUTS] -> par(ninputs, ninputs + ninputs)
        mutable vector<double> ioutput;    // mat[NINPUTS]
        
        /// <summary>
        /// parameters of the fully connected, inner neurons (hidden layer)
        /// </summary>
        matrix<double>::map_type hweights; // mat[NNEURONS][NINPUTS] -> par(2 * ninputs, 2 * ninputs + nneurons * ninputs)
        vector<double>::map_type htheta;   // mat[NNEURONS] -> par(2 * ninputs + nneurons * ninputs, 2 * ninputs + nneurons * ninputs + nneurons)
        mutable vector<double> houtput;    // mat[NNEURONS]
        
        /// <summary>
        /// parameters of the output neurons
        /// </summary>
        matrix<double>::map_type oweights; // mat[NOUTPUTS][NNEURONS] -> par(2 * ninputs + nneurons * ninputs + nneurons, 2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons)
        vector<double>::map_type otheta;   // mat[NOUTPUTS] -> par(2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons, 2 * ninputs + nneurons * ninputs + nneurons + noutputs * nneurons + noutputs)
        mutable vector<double> ooutput;    // mat[NOUTPUTS]

        /// <summary>
        /// 
        /// </summary>
        const size_t ntotparameters, ninputs, noutputs, nneurons;
    } nn;

    /// <summary>
    /// 
    /// </summary>
    typedef struct dataSet {
        /// <summary>
        /// 
        /// </summary>
        dataSet()
            : xx(0), yy(0), 
            ninputs(0), noutputs(0) {}

        /// <summary>
        /// 
        /// </summary>
        dataSet(size_t _ninputs, size_t _noutputs)
            : xx(_ninputs), yy(_noutputs), 
            ninputs(_ninputs), noutputs(_noutputs) {}

        /// <summary>
        /// 
        /// </summary>
        math::vector<double> xx, yy;

        /// <summary>
        /// 
        /// </summary>
        const size_t ninputs, noutputs;
    } dataSet;


    /// <summary>
    /// 
    /// </summary>
    class supervisor {
        public:
            /// <summary>
            /// reset all parameters
            /// </summary>
            static void init(nn& nn) {
                for (int i = 0; i < nn.ntotparameters; ++i)
                    nn.parameters[i] = 0; //rnd(0, 1);
            }

            /// <summary>
            /// 
            /// </summary>
            static void calculateNN(const math::vector<double>& xx, const nn& nn) {
                // TODO: implement flag to choose different transfer functions
                // TODO: define unaryExr in operators definieren
                nn.ioutput = math::eigen::cprod(nn.iweights, xx) - nn.itheta;
                nn.ioutput = nn.ioutput.eigen().unaryExpr(&unarySigmoid);
                
                nn.houtput = nn.hweights * nn.ioutput - nn.htheta;
                nn.houtput = nn.houtput.eigen().unaryExpr(&unarySigmoid);
                
                nn.ooutput = nn.oweights * nn.houtput - nn.otheta;
                nn.ooutput = nn.ooutput.eigen().unaryExpr(&unarySigmoid);
            }

            /// <summary>
            /// train the network (gradient descent method)
            /// </summary>
            static void train(nn& nn, const std::vector<dataSet>& dataset, const double accuracy, const double learningrate) {
                double h = 0.005;
                math::vector<double> deriv(nn.ntotparameters);
                // optimize the cost function
                double lf = 0;
                do {
                    lf = lossFunction(nn, dataset);
                    //std::cout << "lf  = " << lf << std::endl;
                    for(int i = 0; i < nn.ntotparameters; ++i) {
                        double tempi = nn.parameters[i];
                        nn.parameters[i] = nn.parameters[i] + h;
                        struct nn nnhi(nn.parameters, nn.ninputs, nn.noutputs, nn.nneurons);
                        double lfi = lossFunction(nnhi, dataset);
                        //std::cout << "lf" << i << " = " << lf << std::endl;
                        deriv[i] = (lfi - lf) / h;
                        nn.parameters[i] = tempi;
                    }

                    /*
                    std::cout << "param = ";
                    for(int i = 0; i < nn.ntotparameters; ++i)
                         std::cout << nn.parameters[i] << " ";
                    std::cout << std::endl;

                    std::cout << "deriv = ";
                    for(int i = 0; i < nn.ntotparameters; ++i)
                         std::cout << deriv[i] << " ";
                    std::cout << std::endl;
                    */

                    double alpha = learningrate;;
                    for(int i = 0; i < nn.ntotparameters; ++i) {
                        nn.parameters[i] = nn.parameters[i] - alpha * deriv[i];
                    }
                } while (lf > accuracy);
            }


        private:
            /// <summary>
            /// unary transfer function
            /// </summary>
            static double unaryRelu(double x) {
                if (x >= 0) 
                    return x;
                return  0;
            }

            static double unarySigmoid(double x) {
                return 1 / (1 + exp(-x));
            }

            /// <summary>
            /// unary transfer function
            /// </summary>
            static double lossFunction(const nn& nn, const std::vector<dataSet>& dataset) {
                double delta = 0;
                for(int i = 0; i < dataset.size(); ++i) {
                    calculateNN(dataset[i].xx, nn);
                    double delta2 = 0;
                    for(int j = 0; j < dataset[i].yy.size(); ++j) {
                        delta2 += std::pow(nn.ooutput[j] - dataset[i].yy[j], 2);
                    }
                    delta += std::sqrt(delta2);
                }
                return delta / 2;
            }

            /// <summary>
            /// 
            /// </summary>
            static double rnd(double a, double b){
                double x = (double)rand() / (double)(RAND_MAX) * (b - a) + a;
                return x;
            }
            
    };
}