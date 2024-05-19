#!/usr/bin/env python3
"""
    Optimization
"""


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set """
    mov_avg = []
    fv = 0
    for i in range(1, len(data) + 1):
        fv = beta * fv + (1 - beta) * data[i - 1]
        mov_avg.append(fv / (1 - beta ** i))
    return mov_avg
