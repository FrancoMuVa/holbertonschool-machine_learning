#!/usr/bin/env python3
"""
Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    " Performs pooling on images "
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_output = (h - kh) // sh + 1
    w_output = (w - kw) // sw + 1
    convolution = np.zeros((m, h_output, w_output, c))
    imgs = np.arange(m)
    for h2 in range(h_output):
        for w2 in range(w_output):
            img_slct = images[:, h2 * sh:h2 * sh + kh, w2 * sw:w2 * sw + kw]
            if mode == 'max':
                convolution[imgs, h2, w2, :] = np.max(img_slct, axis=(1, 2))
            elif mode == 'avg':
                convolution[imgs, h2, w2, :] = np.mean(img_slct, axis=(1, 2))
    return convolution
