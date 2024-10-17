#!/usr/bin/env python3
"""
Initialize Yolo
"""
from tensorflow import keras as K


class Yolo():
    """ Class Yolo """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Initializes a new instance of Yolo """
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
