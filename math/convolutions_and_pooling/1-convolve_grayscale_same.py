#!/usr/bin/env python3
"""
Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    " Performs a same convolution on grayscale images "
    m, h, w, = images.shape
    kh, kw = kernel.shape
    h_pad, w_pad = kh // 2, kw // 2
    pad_img = np.pad(images, ((0, 0), (h_pad, h_pad), (w_pad, w_pad)))
    convolution = np.zeros((m, h, w))
    imgs = np.arange(m)
    for h2 in range(h):
        for w2 in range(w):
            img_slct = pad_img[:, h2:h2 + kh, w2:w2 + kw]
            convolution[imgs, h2, w2] = np.sum(img_slct * kernel, axis=(1, 2))
    return convolution
