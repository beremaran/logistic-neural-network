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

import os
import argparse

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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('action')

    args = argparser.parse_args()

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

    if args.action == "evaluate":
        for model in models:
            print()
            print('-' * 16)
            print("Model '{}'".format(model["name"]))
            print('-' * 16)
            nn.load_model(os.path.join(MODEL_DIR, model["name"]))
            output = nn.evaluate(model["data"]["x"])
            dump_evaluations(model["data"]["x"], model["data"]["y"], output)
    elif args.action == "train":
        for model in models:
            print()
            print("Training '{}' model ...".format(model["name"]))
            nn.reset_model(1024, model["data"]["x"][0].shape[0])
            output = nn.train(model["data"]["x"], model["data"]["y"], iterations=10000, learning_rate=0.05)
            nn.dump_model(os.path.join(MODEL_DIR, model["name"]))
