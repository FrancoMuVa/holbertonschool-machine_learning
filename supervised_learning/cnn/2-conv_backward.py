#!/usr/bin/env python3
"""
Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    pad_A = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev = np.zeros(pad_A.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    db[:, :, 0, :] = np.sum(dZ, axis=(0, 1, 2))
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    dA_prev[i, h * sh:h * sh + kh, w * sw:w * sw + kw] +=\
                        W[:, :, :, k] * dZ[i, h, w, k]
                    slct = pad_A[i, h * sh:h * sh + kh, w * sw:w * sw + kw]
                    dW[:, :, :, k] += (slct * dZ[i, h, w, k])
    return dA_prev, dW, db
