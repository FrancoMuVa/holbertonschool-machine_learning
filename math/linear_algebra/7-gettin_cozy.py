#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    "Function that concatenates two matrices along a specific axis"
    if (axis == 0) and (len(mat1[0]) == len(mat2[0])):
        new = []
        new = [r1[:] for r1 in mat1] + [r2 for r2 in mat2]
        return new
    elif (axis == 1) and (len(mat1) == len(mat2)):
        new = []
        for row1, row2 in zip(mat1, mat2):
            new.append(row1[:] + row2[:])
        return new
    return None
