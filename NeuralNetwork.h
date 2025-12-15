#pragma once
#include "Layer.h"

class NeuralNetwork {
public:
    Layer hidden;
    Layer output;

    NeuralNetwork(int input, int hidden_size, int output_size);

    double forward(const std::vector<double>& x);
    void backward(const std::vector<double>& x, double y);
    void step(double lr);
};
