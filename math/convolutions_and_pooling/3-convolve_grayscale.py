#!/usr/bin/env python3
"""
Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    " Performs a convolution on grayscale images "
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph, pw = (h - 1) * sh + kh - h // 2, (w - 1) * sw + kw - w // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    _, h_pad, w_pad = pad_img.shape
    h_output = (h_pad - kh + ph * 2) // sh + 1
    w_output = (w_pad - kw + pw * 2) // sw + 1
    convolution = np.zeros((m, h_output, w_output))
    imgs = np.arange(m)
    for h2 in range(h_output):
        for w2 in range(w_output):
            img_slct = pad_img[:, h2 * 2:(h2 * sh) + kh, w2 * 2:(w2 * sw) + kw]
            convolution[imgs, h2, w2] = np.sum(img_slct * kernel, axis=(1, 2))
    return convolution
