#!/usr/bin/env python3
"""
    Function that calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """ Return the derivative of a polynomial """
    if not poly or not isinstance(poly, list):
        return None
    if len(poly) <= 1:
        return [0]
    index = []
    for i in range(len(poly)):
        index.append(i)
    return_list = []
    for i in range(1, len(poly)):
        return_list.append(index[i] * poly[i])
    return return_list
