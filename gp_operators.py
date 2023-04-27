"""
This file contains all genetic operators from Function set.
"""

import numpy as np

# ========================
# Upload typings.
# ========================
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_filter import GPFilter
from gp_utils.gp_size import GPSize
from gp_utils.gp_point import GPPoint
from gp_terminals.gp_cutshape import GPCutshape

# ========================
# Classification operators.
# ========================

def add(a: float, b: float) -> float:
    return a + b

def sub(a: float, b: float):
    return a - b

def mul(a: float, b: float) -> float:
    return a * b

# Protected division
def div(a: float, b: float) -> float:
    if b == 0:
        return 1.0
    else:
        return a / b

# ========================
# Aggregation operators.
# ========================

def agg_mean(image: GPImage, top_left_corner: GPPoint, size: GPSize, shape: GPCutshape) -> float:
    return np.mean(shape.cut(image, top_left_corner, size))

def agg_stdev(image: GPImage, top_left_corner: GPPoint, size: GPSize, shape: GPCutshape) -> float:
    return np.std(shape.cut(image, top_left_corner, size))

def agg_min(image: GPImage, top_left_corner: GPPoint, size: GPSize, shape: GPCutshape) -> float:
    return np.min(shape.cut(image, top_left_corner, size))

def agg_max(image: GPImage, top_left_corner: GPPoint, size: GPSize, shape: GPCutshape) -> float:
    return np.max(shape.cut(image, top_left_corner, size))

# ========================
# Convolutional operators.
# ========================

def conv(image: GPImage, filter: GPFilter) -> GPImage:
    new_image = image.apply_filter(filter)
    #ReLU
    new_image.pixel_data = new_image.pixel_data * (new_image.pixel_data > 0)
    return 

def pool(image: GPImage) -> GPImage:
    return image.apply_maxpool2x2()