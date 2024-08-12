#!/usr/bin/env python3
"""
    Function that calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    " calculate the derivative of polynomial "
    if not poly or not isinstance(poly, list):
        return None
    if len(poly) <= 1:
        return [0]
    idx = []
    for i in range(len(poly)):
        idx.append(i)
    return_list = []
    for i in range(1, len(poly)):
        return_list.append(idx[i] * poly[i])
    return return_list
