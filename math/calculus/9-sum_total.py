#!/usr/bin/env python3
"""
    Function that calculate the sum of squeres from 1 to n.
"""


def summation_i_squared(n):
    " Retutn the sum of squeres from 1 to n "
    return sum(i ** 2 for i in range(1, n + 1))
