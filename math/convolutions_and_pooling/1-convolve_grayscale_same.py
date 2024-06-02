#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Function that performs a same convolution on grayscale images """
    m, he, wi = images.shape
    kh, kw = kernel.shape
    h_pad = kh // 2
    w_pad = kw // 2
    pad_img = np.pad(images, ((0, 0), (h_pad, h_pad), (w_pad, w_pad)))
    comv = np.zeros((m, he, wi))
    img = np.arange(m)
    for h in range(he):
        for w in range(wi):
            sect = pad_img[:, h: h + kh, w: w + kw]
            comv[img, h, w] = np.sum(sect * kernel, axis=(1, 2))
    return comv
