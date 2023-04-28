import numpy as np

from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage
class GPTree:
    def __init__(self, pset, min_depth: int = 2, max_depth: int = 10, tree: gp.PrimitiveTree = None) -> None:
        self.pset = pset
        if tree is None:
            expr = gp.genFull(pset, min_=min_depth, max_=max_depth)
            self.tree = gp.PrimitiveTree(expr)
        else:
            self.tree = tree
        
    def feed(self, image: GPImage) -> float:
        f = gp.compile(self.tree, self.pset)
        x = f(image)
        # try:
        #     x = gp.compile(self.tree, self.pset)(image)
        # except:
        #     print("ERROR!!!")
        #     print(f"{str(self.tree)}")
        return self.sigmoid_activation(x)
    
    def sigmoid_activation(self, x):
        return 1/(1 + np.exp(-x))