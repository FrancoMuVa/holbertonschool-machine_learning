#!/usr/bin/env python3
"""
Convolutional Neural Networks
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer
    of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    pad_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    h_out = (h_prev + ph * 2 - kh) // sh + 1
    w_out = (w_prev + pw * 2 - kw) // sw + 1
    conv = np.zeros((m, h_out, w_out, c_new))

    for h in range(h_out):
        for w in range(w_out):
            sect = pad_img[:, h * sh:((h * sh) + kh),
                           w * sw:((w * sw) + kw), :]
            for c in range(c_new):
                conv[:, h, w, c] = np.sum(sect * W[:, :, :, c], axis=(1, 2, 3))
    conv += b
    return activation(conv)
