import numpy as np

from tree_runner import run_tree
from deap import gp
from gp_terminals.gp_image import GPImage

class GPTree:
    def __init__(self, pset, min_depth: int = 2, max_depth: int = 10, tree: gp.PrimitiveTree = None) -> None:
        self.pset = pset
        if tree is None:
            expr = gp.genFull(pset, min_=min_depth, max_=max_depth)
            self.tree = gp.PrimitiveTree(expr)
        else:
            self.tree = tree
        
    def predict(self, image: GPImage) -> float:
        x = run_tree(image, self.tree)
        return self.sigmoid_activation(x)
    
    def sigmoid_activation(self, x):
        return 1/(1 + np.exp(-x))