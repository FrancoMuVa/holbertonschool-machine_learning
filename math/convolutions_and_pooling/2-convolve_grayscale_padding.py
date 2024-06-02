#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale
    images with custom padding.
    """
    m, he, wi = images.shape
    kh, kw = kernel.shape
    ph = padding[0]
    pw = padding[1]
    h_out = he + (ph * 2) - kh + 1
    w_out = wi + (pw * 2) - kw + 1

    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))

    comv = np.zeros((m, h_out, w_out))
    img = np.arange(m)
    for h in range(h_out):
        for w in range(w_out):
            sect = pad_img[:, h: h + kh, w: w + kw]
            comv[img, h, w] = np.sum(sect * kernel, axis=(1, 2))
    return comv
