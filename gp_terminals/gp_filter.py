import numpy as np
from gp_utils.gp_size import GPSize

class GPFilter:
    """
    Class that contains filter and operations over it.
    """
    
    def __init__(self, filter_data: np.ndarray) -> None:
        """
        Creates filter from numpy array.

        Parameter
        ---------
        filter_data: np.ndarray
            Numpy array of values in range [-3, 3] and shape of (height, width)
        """

        if filter_data.shape[0] != filter_data.shape[1]:
            raise Exception(f"Filter must be a square matrix. Got marix with size {filter_data.shape}")

        if filter_data.shape[0] % 2 == 0:
            raise Exception(f"Filter shape must be odd numer. Got shape {filter_data.shape}")

        self.filter_data = filter_data
        self.size = filter_data.shape[0]

    def apply_to_matrix(self, matrix: np.ndarray) -> float:
        """
        Elementwise multiplication of filter matrix and some other matrix
        and then aggregation of all values by sum.
        """
        
        if self.filter_data.shape != matrix.shape:
            raise Exception(f"""
                Try to apply filter to matrix with different shape.\n
                Expected: {self.filter_data.shape}; Got: {matrix.shape};
            """)

        return np.sum(np.multiply(self.filter_data, matrix))
    
    def __str__(self) -> str:
        np_str = str(self.filter_data.tolist())#.replace('  ', ' ').replace('\n ',',').replace(' ', ',')
        return f"GPFilter(np.array({np_str}))"
    
    __repr__ = __str__