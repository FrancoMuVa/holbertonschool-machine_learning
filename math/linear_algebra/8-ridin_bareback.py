#!/usr/bin/env python3
"""
    Function that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """ Return a new matrix """
    if len(mat1[0]) == len(mat2):
        matrix = []
        x = 0
        for row1 in mat1:
            resu = []
            for col in range(0, len(mat2[0])):
                row1_idx = 0
                val = 0
                for row2 in mat2:
                    val += (row1[row1_idx] * row2[col])
                    row1_idx += 1
                resu.append(val)
            matrix.append(resu)
            x += 1
        return matrix
    else:
        return None
