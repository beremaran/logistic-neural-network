import numpy as np


class Activation:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def d_sigmoid(z):
        return Activation.sigmoid(z) * (1 - Activation.sigmoid(z))

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def relu(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def d_relu(z):
        z[z < 0] = 0
        z[z >= 0] = 1
        return z

    @staticmethod
    def leaky_relu(z):
        z[z < 0.01 * z] = 0.01 * z
        return z
