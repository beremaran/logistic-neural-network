#!/usr/bin/env python

import numpy as np

logic_and = {
    "x": np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]),
    "y": np.array([[0, 0, 0, 1]])
}

logic_or = {
    "x": np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]),
    "y": np.array([[0, 1, 1, 1]])
}

logic_xor = {
    "x": np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]),
    "y": np.array([[0, 1, 1, 0]])
}

logic_not = {
    "x": np.array([
        [0],
        [1],
    ]),
    "y": np.array([[1, 0]])
}

logic_implies = {
    "x": np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]),
    "y": np.array([[1, 1, 0, 1]])
}
