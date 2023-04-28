class GPPercentage:
    """
    Represent relative position in percents.
    """
    
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def __str__(self) -> str:
        return f"GPPercentage({self.a},{self.b})"
    
    __repr__ = __str__