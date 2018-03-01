#!/usr/bin/env python

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import numpy as np

from activations import Activation


class FCNN:
    def __init__(self
                 , input_features=2
                 , hidden_layer_units=4
                 # , activation_function=Activation.relu
                 ):
        self.w = [np.array((0, 0))] * 2
        self.b = [np.array((0, 0))] * 2
        self.reset_model(hidden_layer_units, input_features)
        self.activation_function = Activation.relu

    def evaluate(self
                 , x
                 , dump_all=False):

        # input => hidden
        z1 = np.dot(self.w[0], x.T) + self.b[0]
        a1 = self.activation_function(z1)
        # hidden => output
        z2 = np.dot(self.w[1], a1) + self.b[1]
        a2 = Activation.sigmoid(z2)
        if dump_all:
            return a2, z2, a1, z1
        else:
            return a2

    def train(self
              , x
              , y
              , iterations=100
              , learning_rate=0.001):
        m = x.shape[0]
        costs = []
        verbosity_step = iterations / 10
        for current_iteration in range(iterations):
            a2, z2, a1, z1 = self.evaluate(x, dump_all=True)

            cost = FCNN._cost(a2, y)
            costs.append(cost)
            if current_iteration % verbosity_step == 0:
                print("Loss: {:.8f} Epochs: {:-10d}".format(cost, current_iteration))

            dz2 = a2 - y
            dw2 = np.dot(dz2, a1.T) / m
            db2 = np.sum(dz2, axis=1, keepdims=True) / m

            dz1 = np.dot(self.w[1].T, dz2) * Activation.d_relu(z1)
            dw1 = (np.dot(dz1, x)) / m
            db1 = (np.sum(dz1, axis=1, keepdims=True)) / m

            self.w[0] = self.w[0] - learning_rate * dw1
            self.b[0] = self.b[0] - learning_rate * db1
            self.w[1] = self.w[1] - learning_rate * dw2
            self.b[1] = self.b[1] - learning_rate * db2

        return costs

    @staticmethod
    def _cost(yy, y):
        return np.sum((yy - y) ** 2) / y.shape[0]

    def reset_model(self, hidden_layer_units, input_features):
        self.w = [
            np.random.normal(0, 0.1, (hidden_layer_units, input_features)) * 0.01,
            np.random.normal(0, 0.1, (1, hidden_layer_units)) * 0.01
        ]
        self.b = [
            np.random.normal(0, 0.1, (hidden_layer_units, 1)) * 0.01,
            1
        ]

    def dump_model(self, model_path):
        np.savetxt(model_path + "_w_1.model", self.w[0], delimiter=",")
        np.savetxt(model_path + "_b_1.model", self.b[0], delimiter=",")
        np.savetxt(model_path + "_w_2.model", self.w[1], delimiter=",")
        np.savetxt(model_path + "_b_2.model", self.b[1], delimiter=",")

    def load_model(self, model_path):
        self.w[0] = np.genfromtxt(model_path + "_w_1.model", delimiter=",")
        if len(self.w[0].shape) == 1:
            self.w[0] = np.reshape(self.w[0], (self.w[0].shape[0], 1))

        self.b[0] = np.genfromtxt(model_path + "_b_1.model", delimiter=",")
        if len(self.b[0].shape) == 1:
            self.b[0] = np.reshape(self.b[0], (self.b[0].shape[0], 1))

        self.w[1] = np.genfromtxt(model_path + "_w_2.model", delimiter=",")
        self.b[1] = np.genfromtxt(model_path + "_b_2.model", delimiter=",")
