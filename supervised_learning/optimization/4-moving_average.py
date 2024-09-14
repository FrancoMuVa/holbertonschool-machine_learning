#!/usr/bin/env python3
"""
    Moving Average
"""
import numpy as np


def moving_average(data, beta):
    " Calculates the weighted moving average of a data set "
    m_avg, av = [], 0
    for idx, dt in enumerate(data):
        av = beta * av + (1 - beta) * dt
        m_avg.append(av / (1 - beta ** (idx + 1)))
    return m_avg
