import numpy as np
import gp_operators as ops

from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage
class GPTree:
    def __init__(self, min_depth: int, max_depth: int, pset) -> None:
        self.pset = pset
        self.expr = gp.genFull(self.pset, min_=min_depth, max_=max_depth)
        self.tree = gp.PrimitiveTree(self.expr)
    
    def __init__(self, tree: gp.PrimitiveTree):
        self.tree = tree

    def feed(self, image: GPImage) -> float:
        x = gp.compile(self.tree, self.pset)(image)
        return self.sigmoid_activation(x)
    
    def sigmoid_activation(self, x):
        return 1/(1 + np.exp(-x))