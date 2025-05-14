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
#define SIGMOID
//#define RELU
//#define TANH
//#define COMBINED

namespace math {
    // config for adaptive learning (if used)
    typedef struct adaptive {
        bool apply = false;
        double upperThreshold = 1e-1;
        double lowerThreshold = 1e-3;
        double increase = 1e-1;
        double maxnAdapt = 5;
        mutable double save = 0;
        mutable int nAdapt = 0;
    } adaptive;

    /// <summary>
    /// configuration of the neural net
    /// </summary>
    typedef struct config {
        config()
            : adaptive() {}

        config(const adaptive& _adaptive)
            : adaptive(_adaptive) {}

        adaptive adaptive;
    } config;

    /// <summary>
    /// 
    /// </summary>
    typedef struct nn {
        nn(size_t _ninputs, size_t _noutputs, size_t _nneurons, const config _config = config())
            : parameters(2*_ninputs + _ninputs * _nneurons + _nneurons + _nneurons * _noutputs + _noutputs),
            
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
            ninputs(_ninputs), noutputs(_noutputs), nneurons(_nneurons),

            cconfig(_config) {}

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
        /// number of total parameters, number of inputs, outputs and neurons
        /// </summary>
        const size_t ntotparameters, ninputs, noutputs, nneurons;

        /// <summary>
        /// config that is used to work on the neural net
        /// </summary>
        const config cconfig;
    } nn;

    /// <summary>
    /// A dataset for given inputs and outputs
    /// </summary>
    typedef struct dataSet {
        dataSet()
            : xx(0), yy(0), 
            ninputs(0), noutputs(0) {}

        dataSet(size_t _ninputs, size_t _noutputs)
            : xx(_ninputs), yy(_noutputs), 
            ninputs(_ninputs), noutputs(_noutputs) {}

        /// <summary>
        /// output and input values
        /// </summary>
        math::vector<double> xx, yy;

        /// <summary>
        /// number of inputs and outputs
        /// </summary>
        const size_t ninputs, noutputs;
    } dataSet;

    /// <summary>
    /// Supervisor that trains the network
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
            /// calculate the outputs for a given input
            /// </summary>
            static void calculateNN(const math::vector<double>& xx, const nn& nn) {
                // TODO: put this into the config struct
                #ifdef SIGMOID
                    double (&func)(double) = unarySigmoid;
                #endif

                #ifdef RELU
                    double (&func)(double) = unaryRelu;
                #endif

                #ifdef TANH
                    double (&func)(double) = unaryTanh;
                #endif

                #if defined(SIGMOID) || defined(RELU) || defined(TANH)
                    nn.ioutput = math::eigen::cprod(nn.iweights, xx) - nn.itheta;
                    nn.ioutput = math::eigen::unary(nn.ioutput, &func);

                    nn.houtput = nn.hweights * nn.ioutput - nn.htheta;
                    nn.houtput = math::eigen::unary(nn.houtput, &func);

                    nn.ooutput = nn.oweights * nn.houtput - nn.otheta;
                    nn.ooutput = math::eigen::unary(nn.ooutput, &func);
                #endif

                #ifdef COMBINED
                    nn.ioutput = math::eigen::cprod(nn.iweights, xx) - nn.itheta;
                    nn.ioutput = math::eigen::unary(nn.ioutput, &unarySigmoid);

                    nn.houtput = nn.hweights * nn.ioutput - nn.htheta;
                    nn.houtput = math::eigen::unary(nn.houtput, &unarySigmoid);

                    nn.ooutput = nn.oweights * nn.houtput + nn.otheta;
                    nn.ooutput = math::eigen::unary(nn.ooutput, &unaryRelu);
                #endif
            }

            /// <summary>
            /// train the network (gradient descent method)
            /// </summary>
            static void train(nn& nn, const std::vector<dataSet>& dataset, const double accuracy, const double learningrate) {
                double h = 0.005;
                math::vector<double> deriv(nn.ntotparameters);
                size_t counter = 0;
                // optimize the cost function
                double lf = 0;
                do {
                    lf = lossFunction(nn, dataset);

                    for (int i = 0; i < nn.ntotparameters; ++i) {
                        double tempi = nn.parameters[i];
                        nn.parameters[i] = nn.parameters[i] + h;
                        double lfi = lossFunction(nn, dataset);
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

                    // adapt the parameters
                    double alpha = learningrate;

                    if (nn.cconfig.adaptive.apply) {
                        auto& save = nn.cconfig.adaptive.save;
                        auto& lowerThreshold = nn.cconfig.adaptive.lowerThreshold;
                        auto& upperThreshold = nn.cconfig.adaptive.upperThreshold;
                        auto& nAdapt = nn.cconfig.adaptive.nAdapt;
                        auto& maxnAdapt = nn.cconfig.adaptive.maxnAdapt;
                        auto& increase = nn.cconfig.adaptive.increase;
                        // if there is only a small change in the lossfunction during
                        // two subsequent iterations, increase the learning rate, else decrease it
                        if(std::abs(lf - save) < lowerThreshold) nAdapt++;
                        if(std::fabs(lf - save) > upperThreshold) nAdapt--;
                        if (nAdapt < -maxnAdapt) nAdapt = -maxnAdapt;
                        if (nAdapt > maxnAdapt) nAdapt = maxnAdapt;
                        double fac = std::pow(1 + increase, nAdapt);
                        alpha = alpha * fac;
                        save = lf;
                    }
                    //std::cout << alpha << std::endl;
                    nn.parameters -= (alpha * deriv);

                    // Status
                    if (counter++ % 100 == 0)
                        std::cout << "lf  = " << lf << std::endl;
                } while (lf > accuracy);
            }


        private:
            /// <summary>
            /// unary relu transfer function
            /// </summary>
            static double unaryRelu(double x) {
                if (x >= 0) 
                    return x;
                return  0;
            }

            /// <summary>
            /// unary sigmoid transfer function
            /// </summary>
            static double unarySigmoid(double x) {
                return 1 / (1 + std::exp(-x));
            }

            /// <summary>
            /// unary tanh transfer function
            /// </summary>
            static double unaryTanh(double x) {
                return std::tanh(x);
            }

            /// <summary>
            /// loss function
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
            /// random number generator
            /// </summary>
            static double rnd(double a, double b){
                double x = (double)rand() / (double)(RAND_MAX) * (b - a) + a;
                return x;
            }
            
    };
}