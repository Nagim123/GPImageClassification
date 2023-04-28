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
        self.x = int(x)
        self.y = int(y)

    def __str__(self) -> str:
        return f"GPPoint({self.x},{self.y})"
    
    __repr__ = __str__