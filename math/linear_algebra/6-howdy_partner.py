#!/usr/bin/env python3
"""
    Function that concatenates two arrays.
"""


def cat_arrays(arr1, arr2):
    """ Return a new array """
    new_arr = arr1[:]
    new_arr.extend(arr2)
    return new_arr
