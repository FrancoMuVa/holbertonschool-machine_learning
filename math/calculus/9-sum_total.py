#!/usr/bin/env python3
"""
    Function that that calculates sum_{i=1}^{n} i^2.
"""


def summation_i_squared(n):
    """ Return a integer """
    if n <= 0:
        return None
    elif n == 1:
        return 1
    else:
        return n ** 2 + summation_i_squared(n - 1)
