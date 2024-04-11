#!/usr/bin/env python3
"""
    Function that concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Return a new matrix """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        matrix = [row[:] for row in mat1]
        for row2 in mat2:
            matrix.append(row2[:])
        return matrix
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        matrix = []
        for row1, row2 in zip(mat1, mat2):
            matrix.append(row1[:] + row2[:])
        return matrix
    return None
