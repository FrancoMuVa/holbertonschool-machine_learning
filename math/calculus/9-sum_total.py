#!/usr/bin/env python3
"""
    Function that that calculates sum_{i=1}^{n} i^2.
"""


def summation_i_squared(n):
    """ Return a integer """
    if not isinstance(n, int):
        return None
    return ((n * ((n + 1) * (2 * n + 1))) // (6))
