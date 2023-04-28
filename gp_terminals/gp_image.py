import numpy as np
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_filter import GPFilter

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
        
        f = filter.size
        pixel_data = np.pad(self.pixel_data, (f - 1)//2)
        
        result = np.zeros(self.pixel_data.shape)
        for i in range(pixel_data.shape[0] - f + 1):
            for j in range(pixel_data.shape[1] - f + 1):
                result[i][j] = filter.apply_to_matrix(pixel_data[i:i+f, j:j+f])

        return GPImage(result)
    
    def apply_maxpool2x2(self) -> object:
        if self.size.h % 2 != 0 or self.size.w % 2 != 0:
            raise Exception(f"To apply max pool image's size must be divisible by two. Shape:{self.size.shape()}")
        
        result = np.zeros((self.size.h//2, self.size.w//2))
        for i in range(self.size.h - 1):
            for j in range(self.size.w - 1):
                result[i][j] = np.max(self.pixel_data[i:i+1, j:j+1])
        
        return GPImage(result)