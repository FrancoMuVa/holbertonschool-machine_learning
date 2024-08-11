#!/usr/bin/env python3
"""
    Function that calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    " Calculates the integral of a polynomial "
    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None
    return_list = [C]
    if len(poly) == 1:
        if poly[0] == 0:
            return [0]
        return_list.append(poly[0])
        return return_list
    for idx, cff in enumerate(poly):
        integral_coeff = cff / (idx + 1)
        if integral_coeff.is_integer():
            integral_coeff = int(integral_coeff)
        return_list.append(integral_coeff)
    return return_list
