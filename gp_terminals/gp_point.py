class GPPoint:
    """
    Container for 2D point.
    """
    def __init__(self, x: int, y: int) -> None: 
        """
        Creates point object from x and y coodrinates.

        Parameter
        ---------
        x: int
            X coordinate.
        y: int
            Y coordinate.
        """ 
        self.x = x
        self.y = y