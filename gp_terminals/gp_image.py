import numpy as np
from gp_utils.gp_size import GPSize
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
        if self.size.h == 1 and self.size.w == 1:
            return GPImage(self.pixel_data.copy())
        result = np.zeros((self.size.h//2, self.size.w//2))
        for i in range(self.size.h - 1, 2):
            for j in range(self.size.w - 1, 2):
                result[i//2][j//2] = np.max(self.pixel_data[i:i+1, j:j+1])
        
        return GPImage(result)
    
    def __str__(self) -> str:
        np_str = str(self.pixel_data.tolist())#.replace('  ', ' ').replace('\n ',',').replace(' ', ',')
        return f"GPImage(np.array({np_str}))"
    
    __repr__ = __str__