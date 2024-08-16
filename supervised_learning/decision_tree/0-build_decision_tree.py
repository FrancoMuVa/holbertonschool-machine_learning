#!/usr/bin/env python3
"""
    Ddecition tree
"""
import numpy as np

class Node:
    """ Node class """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        " Initializes a new instance of Node "
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self) :
        " Returns the maximun depths of the nodes "
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return left if left >= right else right

class Leaf(Node):
    """ Leaf class """
    def __init__(self, value, depth=None):
        """ Initializate a new instance of Leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        " Return the depth "
        return self.depth

class Decision_Tree():
    """ Decision Tree class """
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        """ Initializate a new instance of Decision Tree """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self) :
        " Return the depth "
        return self.root.max_depth_below()
