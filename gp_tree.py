import numpy as np

from deap import gp
from gp_terminals.gp_image import GPImage
class GPTree:
    def __init__(self, min_depth: int, max_depth: int, pset) -> None:
        self.pset = pset
        expr = gp.genFull(self.pset, min_=min_depth, max_=max_depth)
        self.tree = gp.PrimitiveTree(self.expr)
    
    def __init__(self, tree: gp.PrimitiveTree, pset):
        self.tree = tree
        self.pset = pset
        
    def feed(self, image: GPImage) -> float:
        x = gp.compile(self.tree, self.pset)(image)
        return self.sigmoid_activation(x)
    
    def sigmoid_activation(self, x):
        return 1/(1 + np.exp(-x))