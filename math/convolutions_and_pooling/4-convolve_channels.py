#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ Function that performs a convolution on images with channels """
    m, he, wi, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == "same":
        ph = ((he - 1) * sh + kh - he) // 2 + 1
        pw = ((wi - 1) * sw + kw - wi) // 2 + 1
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    h_out = (he + ph * 2 - kh) // sh + 1
    w_out = (wi + pw * 2 - kw) // sw + 1

    comv = np.zeros((m, h_out, w_out))
    img = np.arange(m)

    for h in range(h_out):
        for w in range(w_out):
            sect = pad_img[:, h * sh:((h * sh) + kh), w * sw:((w * sw) + kw)]
            comv[img, h, w] = np.sum(sect * kernel, axis=(1, 2, 3))
    return comv
