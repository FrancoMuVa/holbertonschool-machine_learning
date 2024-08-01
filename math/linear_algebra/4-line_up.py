#!/usr/bin/env python3
"""
    Function that adds two arrays element-wise
"""

def add_arrays(arr1, arr2):
    "Return a new matrix"
    if len(arr1) != len(arr2):
        return None
    new = []
    for i in range(0, len(arr1)):
        new.append(arr1[i] + arr2[i])
    return new
