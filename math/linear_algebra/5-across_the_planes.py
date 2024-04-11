#!/usr/bin/env python3
""" Function that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ Returns the sum of two matrices element-wise """
    length = len(mat1[0]) == len(mat2[0])
    if length:
        matrix = []
        for row_1, row_2 in zip(mat1, mat2):
            row = []
            for i in range(0, len(row_1)):
                row.append(row_1[i] + row_2[i])
            matrix.append(row)
        return matrix
    elif length:
        return None
