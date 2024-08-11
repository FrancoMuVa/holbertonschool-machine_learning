#!/usr/bin/env python3
"""
    Function that calculate the sum of squeres from 1 to n.
"""


def summation_i_squared(n):
    " Retutn the sum of squeres from 1 to n "
    return (n * (n + 1) * (2 * n + 1)) // 6 if isinstance(n, int) and n >= 1\
        else None
