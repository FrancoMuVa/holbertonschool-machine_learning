#!/usr/bin/env python3
"""
Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    " Performs forward propagation over a pooling layer of a neural network "
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_output = (h_prev - kh) // sh + 1
    w_output = (w_prev - kw) // sw + 1
    conv = np.zeros((m, h_output, w_output, c_prev))
    for i in range(h_output):
        for j in range(w_output):
            slct = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                conv[:, i, j, :] = np.max(slct, axis=(1, 2))
            elif mode == 'avg':
                conv[:, i, j, :] = np.mean(slct, axis=(1, 2))
    return conv
