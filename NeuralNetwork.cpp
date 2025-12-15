#include "NeuralNetwork.h"
#include <cmath>

NeuralNetwork::NeuralNetwork(int input, int hidden_size, int output_size)
    : hidden(input, hidden_size),
      output(hidden_size, output_size)
{}

double NeuralNetwork::forward(const std::vector<double>& x) {
    auto h = hidden.forward(x);
    auto y_hat = output.forward(h);
    return y_hat[0];
}

void NeuralNetwork::backward(const std::vector<double>& x, double y) {
    double y_hat = output.a[0];
    double loss_grad = 2.0 * (y_hat - y); // dMSE/dy_hat

    output.delta[0] = loss_grad *
                      (output.a[0] * (1 - output.a[0]));

    // Gradients for output layer
    for (int j = 0; j < hidden.output_size; ++j)
        output.dW[0][j] = output.delta[0] * hidden.a[j];
    output.db[0] = output.delta[0];

    hidden.backward(x, output.delta, output.W);
}

void NeuralNetwork::step(double lr) {
    hidden.update(lr);
    output.update(lr);
}
