#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c):
                    if mode == 'max':
                        slct = a_prev[h * sh:(h * sh) + kh,
                                      w * sw:(w * sw) + kw, k]
                        mask = (slct == np.max(slct))
                        dA_prev[i, h * sh:(h * sh) + kh, w * sw:(w * sw) + kw,
                                k] += np.multiply(mask, dA[i, h, w, k])
                    elif mode == 'avg':
                        dA_avg = dA[i, h, w, k] / (kh * kw)
                        dA_prev[i,
                                h * sh:h * sh + kh,
                                w * sw:w * sw + kw,
                                k] += np.ones((kh, kw)) * dA_avg
        return dA_prev
