import numpy as np

class GPSize:
    """
    Container for pair of width and height.
    """
    def __init__(self, w: int, h: int) -> None:
        """
        Creates size object from pair of weight and height.

        Parameter
        ---------
        w: int
            Width of size.
        h: int
            Height of size.
        """ 
        self.w = w
        self.h = h
    
    def shape(self) -> np._ShapeType:
        return np.shape([self.h, self.w])