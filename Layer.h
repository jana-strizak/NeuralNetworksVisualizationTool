#pragma once
#include <vector>

class Layer {
public:
    int input_size;
    int output_size;

    // Parameters
    std::vector<std::vector<double>> W;
    std::vector<double> b;

    // Forward pass
    std::vector<double> z;  // pre-activation
    std::vector<double> a;  // activation

    // Backward pass
    std::vector<double> delta;                 // dL/dz
    std::vector<std::vector<double>> dW;       // dL/dW
    std::vector<double> db;                    // dL/db

    Layer(int in, int out);

    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& prev_a,
                                  const std::vector<double>& next_delta,
                                  const std::vector<std::vector<double>>& next_W);

    void update(double lr);
};
