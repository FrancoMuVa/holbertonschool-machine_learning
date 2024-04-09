#!/usr/bin/env python3

def matrix_shape(matrix):
    row = len(matrix)
    if isinstance(matrix[0], list):
        col = len(matrix[0])
        if isinstance(matrix[0][0], list):
            sli = len(matrix[0][0])
            return [row, col, sli]
        return [row, col]
    return [row]
