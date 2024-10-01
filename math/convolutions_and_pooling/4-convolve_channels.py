#!/usr/bin/env python3
"""
Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    " Performs a convolution on images with channels "
    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    h_output = (h + ph * 2 - kh) // sh + 1
    w_output = (w + pw * 2 - kw) // sw + 1
    convolution = np.zeros((m, h_output, w_output))
    imgs = np.arange(m)
    for h2 in range(h_output):
        for w2 in range(w_output):
            img_slct = pad_img[:, h2 * sh:h2 * sh + kh, w2 * sw:w2 * sw + kw]
            convolution[imgs, h2, w2] = np.sum(img_slct * kernel, (1, 2, 3))
    return convolution
