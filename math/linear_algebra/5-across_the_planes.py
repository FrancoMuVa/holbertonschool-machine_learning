#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    "Function that adds two matrices element-wise"
    if len(mat1[0]) != len(mat2[0]):
        return None
    new = []
    for m1_row, m2_row in zip(mat1, mat2):
        row = []
        for i in range(0, len(mat1[0])):
            row.append(m1_row[i] + m2_row[i])
        new.append(row)
    return new
