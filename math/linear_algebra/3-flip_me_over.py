#!/usr/bin/env python3
def matrix_transpose(matrix):
    mat = []
    i = 0
    max_len = len(matrix[0])
    for _ in range(max_len):
        row = []
        for e in matrix:
            row.append(e[i])
        mat.append(row)
        i += 1
    return mat
