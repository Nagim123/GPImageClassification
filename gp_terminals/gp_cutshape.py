import numpy as np

from gp_terminals.gp_image import GPImage
from gp_utils.gp_size import GPSize
from gp_utils.gp_point import GPPoint

class GPCutshape:
    """
    Special terminal to make shaped-cut from image.
    """
    
    def __init__(self, type_name: str) -> None:
        """
        Initialize Cutshape.

        Parameter
        ---------
        type_name: str
            Type of cut: 'elp', 'row', 'col', 'rec'.
        """
        if not type_name in ['elp', 'row', 'col', 'rec']:
            raise Exception(f"Undefined cut type {type_name}")

        self.type_name = type_name

    def cut(self, image: GPImage, top_left_corner: GPPoint, size: GPSize) -> np.ndarray:
        """
        Get all values inside a shape.

        Parameter
        ---------
        image: GPImage
            The image.
        """
        
        x, y = top_left_corner.x, top_left_corner.y
        w, h = size.w, size.h
        if self.type_name == 'elp':
            pixel_data = image.pixel_data
            mx, my = np.meshgrid(np.arange(pixel_data.shape[1]), np.arange(pixel_data.shape[0]))
            if w == 0 or h == 0:
                return pixel_data.copy()
            ellipse_mask = ((mx - x) / w)**2 + ((my - y) / h)**2 <= 1
            return pixel_data[ellipse_mask].copy()
        elif self.type_name == 'row':
            return image.pixel_data[y, x:max(x+w, image.size.w)].copy()
        elif self.type_name == 'col':
            return image.pixel_data[y:max(y+h, image.size.h), x].copy()
        elif self.type_name == 'rec':
            return image.pixel_data[y:max(y+h, image.size.h), x:max(x+w, image.size.w)].copy()
    def __str__(self) -> str:
        return f"GPCutshape('{self.type_name}')"
    
    __repr__ = __str__