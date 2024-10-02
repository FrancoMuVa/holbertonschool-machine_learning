#!/usr/bin/env python3
"""
Convolutional Forward Prop
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional
    layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    pad_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    h_output = (h_prev + ph * 2 - kh) // sh + 1
    w_output = (w_prev + pw * 2 - kw) // sw + 1
    conv = np.zeros((m, h_output, w_output, c_new))
    for i in range(h_output):
        for j in range(w_output):
            slct = pad_img[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            conv[:, i, j, :] = np.tensordot(
                slct, W, axes=((1, 2, 3), (0, 1, 2))) + b
    return activation(conv)
