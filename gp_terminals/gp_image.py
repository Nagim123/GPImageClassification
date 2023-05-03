import numpy as np
from gp_utils.gp_size import GPSize
from gp_terminals.gp_filter import GPFilter
from scipy import signal

class GPImage:
    """
    Custom class for working with image data.
    """
    
    def __init__(self, pixel_data: np.ndarray) -> None:
        """
        Creates image from numpy array.

        Parameter
        ---------
        pixel_data: np.ndarray
            Numpy array of pixels in range [0; 255] and shape of (height, width)
        """

        self.pixel_data = pixel_data
        self.size = GPSize(pixel_data.shape[1], pixel_data.shape[0])
    
    def apply_filter(self, filter: GPFilter) -> object:
        """
        Apply kernel filter to image by convolution process.
        """
        return GPImage(signal.convolve(self.pixel_data, filter.filter_data, mode='same'))
    
    def apply_maxpool2x2(self) -> object:
        """
        Apply max pool 2d with size 2x2 and stride equal to 2.
        """
        result = self.pixel_data.copy()
        w, h = self.size.w, self.size.h

        if w == 1 and h == 1:
            return GPImage(result)

        if h % 2 != 0:
            result = result[:-1,:]
        if w % 2 != 0:
            result = result[:,:-1]
        h, w = result.shape
        
        return GPImage(result.reshape(h//2, 2, w//2, 2).max(axis=(1,3)))
    
    def __str__(self) -> str:
        np_str = str(self.pixel_data.tolist())
        return f"GPImage(np.array({np_str}))"
    
    __repr__ = __str__
