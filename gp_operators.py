"""
This file contains all genetic operators from Function set.
"""

import numpy as np

# ========================
# Upload typings.
# ========================
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_percentage import GPPercentage 
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_percentage_size import GPPercentageSize

from gp_utils.gp_point import GPPoint
from gp_utils.gp_size import GPSize

# ========================
# Classification operators.
# ========================

def add(a: float, b: float) -> float:
    return a + b

def sub(a: float, b: float) -> float:
    return a - b

def mul(a: float, b: float) -> float:
    return a * b

# Protected division
def div(a: float, b: float) -> float:
    return 1.0 if b == 0 else a / b

# ========================
# Aggregation operators.
# ========================

def agg_mean(image: GPImage, top_left_corner: GPPercentage, size: GPPercentageSize, shape: GPCutshape) -> float:
    top_left_corner = GPPoint(top_left_corner.a * image.size.w, top_left_corner.b * image.size.h)
    size = GPSize(size.a * image.size.w, size.b * image.size.h)
    return np.mean(shape.cut(image, top_left_corner, size))

def agg_stdev(image: GPImage, top_left_corner: GPPercentage, size: GPPercentageSize, shape: GPCutshape) -> float:
    top_left_corner = GPPoint(top_left_corner.a * image.size.w, top_left_corner.b * image.size.h)
    size = GPSize(size.a * image.size.w, size.b * image.size.h)
    return np.std(shape.cut(image, top_left_corner, size))

def agg_min(image: GPImage, top_left_corner: GPPercentage, size: GPPercentageSize, shape: GPCutshape) -> float:
    top_left_corner = GPPoint(top_left_corner.a * image.size.w, top_left_corner.b * image.size.h)
    size = GPSize(size.a * image.size.w, size.b * image.size.h)
    return np.min(shape.cut(image, top_left_corner, size))

def agg_max(image: GPImage, top_left_corner: GPPercentage, size: GPPercentageSize, shape: GPCutshape) -> float:
    top_left_corner = GPPoint(top_left_corner.a * image.size.w, top_left_corner.b * image.size.h)
    size = GPSize(size.a * image.size.w, size.b * image.size.h)
    return np.max(shape.cut(image, top_left_corner, size))

# ========================
# Convolutional operators.
# ========================

def conv(image: GPImage, filter: GPFilter) -> GPImage:
    new_image = image.apply_filter(filter)
    #ReLU
    new_image.pixel_data = new_image.pixel_data * (new_image.pixel_data > 0)
    return new_image

def pool(image: GPImage) -> GPImage:
    return image.apply_maxpool2x2()