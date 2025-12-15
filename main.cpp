#include "NeuralNetwork.h"
#include <iostream>

int main() {
    NeuralNetwork net(2, 4, 1);

    std::vector<double> x = {1000.0, 0.1}; // intentionally unnormalized
    double y = 1.0;

    for (int epoch = 0; epoch < 100; ++epoch) {
        double y_hat = net.forward(x);
        net.backward(x, y);
        net.step(0.01);

        std::cout << "Epoch " << epoch
                  << " | y_hat = " << y_hat << std::endl;
    }
}
