#!/usr/bin/env python3
""" Function that returns the shape of a matrix """


def matrix_shape(matrix):
    """ Returns the shape of a matrix """
    row = len(matrix)
    if isinstance(matrix[0], list):
        col = len(matrix[0])
        if isinstance(matrix[0][0], list):
            sli = len(matrix[0][0])
            return [row, col, sli]
        return [row, col]
    return [row]
