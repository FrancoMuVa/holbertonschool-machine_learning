#!/usr/bin/env python3
"""
    Ddecition tree
"""
import numpy as np


class Node:
    """ Node class """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        " Initializes a new instance of Node "
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        " Returns the maximun depths of the nodes "
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return left if left >= right else right

    def count_nodes_below(self, only_leaves=False):
        " Returns the numbers of nodes "
        if only_leaves and self.is_leaf:
            return 1
        elif self.is_leaf:
            return 1
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)
        total = left + right
        if not only_leaves:
            return total + 1
        return total

    def left_child_add_prefix(self, text):
        " Left child add prefix "
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        " Right child add prefix "
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x != '':
                new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        " Return a string "
        root_line = ''
        if self.is_root:
            root_line = f'root [feature={self.feature}, \
            threshold={self.threshold}]'
        left = self.left_child_add_prefix(self.left_child.__str__())
        right = self.right_child_add_prefix(self.right_child.__str__())
        if self.is_root:
            return f'{root_line}\n{left}{right}'
        return f'-> node [feature={self.feature}, \
            threshold={self.threshold}]\n{left}{right}'

    def get_leaves_below(self):
        " Returns the list of all leaves of the tree "
        if self.is_leaf:
            return 1
        left = self.left_child.get_leaves_below()
        right = self.right_child.get_leaves_below()
        return left + right

    def update_bounds_below(self):
        " Update bounds below "
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}
        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child == self.left_child:
                child.lower.update({self.feature: self.threshold})
            if child == self.right_child:
                child.upper.update({self.feature: self.threshold})
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        " Update indicator "
        def is_large_enough(x):
            return np.all(np.array([np.greater(
                x[:, key], self.lower[key]) for key in list(
                    self.lower.keys())]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([np.less_equal(
                x[:, key], self.upper[key]) for key in list(
                    self.upper.keys())]), axis=0)
        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf(Node):
    """ Leaf class """
    def __init__(self, value, depth=None):
        """ Initializate a new instance of Leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        " Return the depth "
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        " Counts nodes "
        return 1

    def __str__(self):
        " Return a string "
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        " Returns the leaves "
        return [self]

    def update_bounds_below(self):
        " Update bounds "
        pass


class Decision_Tree():
    """ Decision Tree class """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
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

    def depth(self):
        " Return the depth "
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        " Counts nodes "
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        " Return a string "
        return self.root.__str__()

    def get_leaves(self):
        " Returns the leaves "
        return self.root.get_leaves_below()

    def update_bounds(self):
        " Update bounds "
        self.root.update_bounds_below()
