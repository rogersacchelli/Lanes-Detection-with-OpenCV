import numpy as np

class Line:
    def __init__(self):
        self.right__detected = False
        self.left_inside_detected = False
        self.left_outside_detected = False
        self.right_inside_detected = False
        self.right_out_detected = False
        self.radius = 0.0
        self.turn_side = 0

class ImageLine:
    def __init__(self, image):
        self.shape_h = image.shape[0]
        self.shape_w = image.shape[1]
        self.binary_output = np.zeros_like(image)


