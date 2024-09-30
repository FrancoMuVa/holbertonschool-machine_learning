#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    " Performs a valid convolution on grayscale images "
    m, h, w, = images.shape
    kh, kw = kernel.shape
    convolution = np.zeros((m, h - kh + 1, w - kw + 1))
    imgs = np.arange(m)
    for h2 in range(h - kh + 1):
        for w2 in range(w - kw + 1):
            img_slct = images[:, h2:h2 + kh, w2:w2 + kw]
            convolution[imgs, h2, w2] = np.sum(img_slct * kernel, axis=(1, 2))
    return convolution
