#!/usr/bin/env python3
"""
Initialize Yolo
"""
from tensorflow import keras as K
import numpy as np


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

    @staticmethod
    def sigmoid(x):
        """Applies sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ Processes the YOLO model outputs """
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, _, _ = output.shape
            box = output[..., :4]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            tx_sig = self.sigmoid(t_x)
            ty_sig = self.sigmoid(t_y)
            tx_norm = (tx_sig + np.arange(grid_w).reshape(1, grid_w, 1)) / \
                grid_w
            ty_norm = (ty_sig + np.arange(grid_h).reshape(grid_h, 1, 1)) / \
                grid_h
            bw_norm = (np.exp(t_w) * self.anchors[i, :, 0].reshape(1, 1, -1)) \
                / image_width
            bh_norm = (np.exp(t_h) * self.anchors[i, :, 1].reshape(1, 1, -1)) \
                / image_height

            x1 = tx_norm - bw_norm / 2
            y1 = ty_norm - bh_norm / 2
            x2 = tx_norm + bw_norm / 2
            y2 = ty_norm + bh_norm / 2
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)
            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return (boxes, box_confidences, box_class_probs)
