#!/usr/bin/env python3
"""
    Function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    "Return a new matrix"
    new = []
    for idx in range(len(matrix[0])):
        row = []
        for e in matrix:
            row.append(e[idx])
        new.append([row])
    return new
