#!/usr/bin/env python3
"""
    Function that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    " Return a new matrix "
    if len(mat1[0]) != len(mat2):
        return None
    new = []
    for row1 in mat1:
        row = []
        for i in range(len(mat2[0])):
            val, idx_r1 = 0, 0
            for row2 in mat2:
                val += row1[idx_r1] * row2[i]
                idx_r1 += 1
            row.append(val)
        new.append(row)
    return new
