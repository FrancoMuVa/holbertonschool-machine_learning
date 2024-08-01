#!/usr/bin/env python3

def matrix_shape(matrix):
    "Function that calculates the shape of a matrix"
    shape = []
    new_matrix = matrix
    while (isinstance(new_matrix, list)):
        shape.append(len(new_matrix))
        new_matrix = new_matrix[0]
    return shape
