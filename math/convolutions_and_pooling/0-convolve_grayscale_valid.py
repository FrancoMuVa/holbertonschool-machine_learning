#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Function that performs a valid convolution on grayscale images """
    m, he, wi = images.shape
    kh, kw = kernel.shape
    h_out = he - kh + 1
    w_out = wi - kw + 1
    comv = np.zeros((m, h_out, w_out))
    img = np.arange(m)
    for h in range(h_out):
        for w in range(w_out):
            sect = images[:, h: h + kh, w: w + kw]
            comv[img, h, w] = np.sum(sect * kernel, axis=(1, 2))
    return comv
