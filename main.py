#!/usr/bin/env python3

from __future__ import print_function

import os
import sys

import numpy as np

from fcnn import FCNN
import datasets

MODEL_DIR = "models"


def dump_evaluations(x, y, yy):
    _yy = np.copy(yy)
    yy[yy < 0.5] = 0
    yy[yy >= 0.5] = 1
    yy = np.reshape(yy, (yy.shape[0], 1)).T
    _yy = np.reshape(_yy, (_yy.shape[0], 1)).T
    m = x.shape[0]

    print("Input\tExpected\tGuess")
    for i in range(m):
        print("{}\t{}\t\t{} ({:.4f})".format(x[i], y[0][i], yy[0][i], _yy[0][i]))


if __name__ == "__main__":
    nn = FCNN()
    models = [
        {
            "name": "not",
            "data": datasets.logic_not
        },
        {
            "name": "xor",
            "data": datasets.logic_xor
        },
        {
            "name": "and",
            "data": datasets.logic_and
        },
        {
            "name": "or",
            "data": datasets.logic_or
        },
        {
            "name": "implies",
            "data": datasets.logic_implies
        }
    ]

    if sys.argv[1] == "evaluate":
        for model in models:
            print()
            print('-' * 16)
            print("Model '{}'".format(model["name"]))
            print('-' * 16)
            nn.load_model(os.path.join(MODEL_DIR, model["name"]))
            output = nn.evaluate(model["data"]["x"])
            dump_evaluations(model["data"]["x"], model["data"]["y"], output)
    elif sys.argv[1] == "train":
        for model in models:
            print()
            print("Training '{}' model ...".format(model["name"]))
            nn.reset_model(1024, model["data"]["x"][0].shape[0])
            output = nn.train(model["data"]["x"], model["data"]["y"], iterations=10000, learning_rate=0.05)
            nn.dump_model(os.path.join(MODEL_DIR, model["name"]))
