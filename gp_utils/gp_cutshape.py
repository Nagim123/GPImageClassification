import numpy as np

from gp_image import GPImage
from gp_size import GPSize
from gp_point import GPPoint

class GPCutshape:
    def __init__(self, type_name: str) -> None:
        self.type_name = type_name

    def cut(self, image: GPImage, top_left_corner: GPPoint, size: GPSize) -> np.ndarray:
        """
        Get all values inside a shape.
        """
        
        x, y = top_left_corner.x, top_left_corner.y
        w, h = size.w, size.h
        if self.type_name == 'elp':
            pixel_data = image.pixel_data
            mx, my = np.meshgrid(np.arange(pixel_data.shape[1]), np.arange(pixel_data.shape[0]))
            ellipse_mask = ((mx - x) / w)**2 + ((my - y) / h)**2 <= 1
            return pixel_data[ellipse_mask]
        elif self.type_name == 'row':
            return image.pixel_data[y, x:x+w]
        elif self.type_name == 'col':
            return image.pixel_data[y:y+h, x]
        elif self.type_name == 'rec':
            return image.pixel_data[y:y+h, x:x+w]

