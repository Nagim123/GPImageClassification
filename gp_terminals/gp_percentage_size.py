from gp_terminals.gp_percentage import GPPercentage

class GPPercentageSize:
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def __str__(self) -> str:
        return f"GPPercentageSize({self.a},{self.b})"
    __repr__ = __str__