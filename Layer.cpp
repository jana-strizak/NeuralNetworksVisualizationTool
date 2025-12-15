#include "Layer.h"
#include <cmath>
#include <random>

static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

Layer::Layer(int in, int out)
    : input_size(in), output_size(out)
{
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    W.resize(out, std::vector<double>(in));
    b.resize(out, 0.0);

    z.resize(out);
    a.resize(out);
    delta.resize(out);

    dW.resize(out, std::vector<double>(in));
    db.resize(out);

    for (int i = 0; i < out; ++i)
        for (int j = 0; j < in; ++j)
            W[i][j] = dist(gen) * 0.1;
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    for (int i = 0; i < output_size; ++i) {
        z[i] = b[i];
        for (int j = 0; j < input_size; ++j)
            z[i] += W[i][j] * input[j];

        a[i] = sigmoid(z[i]);
    }
    return a;
}

std::vector<double> Layer::backward(
    const std::vector<double>& prev_a,
    const std::vector<double>& next_delta,
    const std::vector<std::vector<double>>& next_W)
{
    for (int i = 0; i < output_size; ++i) {
        double sum = 0.0;
        for (int k = 0; k < next_delta.size(); ++k)
            sum += next_W[k][i] * next_delta[k];

        delta[i] = sum * sigmoid_derivative(z[i]);
    }

    for (int i = 0; i < output_size; ++i) {
        db[i] = delta[i];
        for (int j = 0; j < input_size; ++j)
            dW[i][j] = delta[i] * prev_a[j];
    }

    return delta;
}

void Layer::update(double lr) {
    for (int i = 0; i < output_size; ++i) {
        b[i] -= lr * db[i];
        for (int j = 0; j < input_size; ++j)
            W[i][j] -= lr * dW[i][j];
    }
}
