#!/usr/bin/env python3
"""
Convolutional Neural Networks
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs forward propagation over a pooling
        layer of a neural network.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    matrix = np.zeros((m, h_out, w_out, c_prev))

    for h in range(h_out):
        for w in range(w_out):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw
            if mode == 'max':
                matrix[:, h, w] = np.max(A_prev[
                    :, h_start:h_end, w_start:w_end
                    ], axis=(1, 2))
            elif mode == 'avg':
                matrix[:, h, w] = np.mean(A_prev[
                    :, h_start:h_end, w_start:w_end
                    ], axis=(1, 2))
    return matrix
