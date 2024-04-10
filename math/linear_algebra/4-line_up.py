#!/usr/bin/env python3
""" Function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """
        Return a new array with the add of arr1 and arr2
    """
    if len(arr1) != len(arr2):
        return None
    new_arr = []
    for i in range(0, len(arr1)):
        new_arr.append(arr1[i] + arr2[i])
    return new_arr
