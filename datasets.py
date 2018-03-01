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
