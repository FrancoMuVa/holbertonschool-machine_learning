#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    "Function that adds two arrays element-wise"
    if len(arr1) != len(arr2):
        return None
    new = []
    for i in range(0, len(arr1)):
        new.append(arr1[i] + arr2[i])
    return new
