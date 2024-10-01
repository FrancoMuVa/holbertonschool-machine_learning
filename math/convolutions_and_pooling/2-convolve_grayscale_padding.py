#!/usr/bin/env python3
"""
Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    " Performs a convolution on grayscale images with custom padding "
    m = images.shape[0]
    kh, kw = kernel.shape
    ph, pw = padding
    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    _, h_pad, w_pad = pad_img.shape
    h_output, w_output = h_pad - kh + 1, w_pad - kw + 1
    convolution = np.zeros((m, h_output, w_output))
    imgs = np.arange(m)
    for h2 in range(h_output):
        for w2 in range(w_output):
            img_slct = pad_img[:, h2:h2 + kh, w2:w2 + kw]
            convolution[imgs, h2, w2] = np.sum(img_slct * kernel, axis=(1, 2))
    return convolution
