#!/usr/bin/env python3
"""
    Function that performs element-wise addition, subtraction, multiplication,
    and division
"""


def np_elementwise(mat1, mat2):
    " Return a tuple containing the element-wise operation "
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
