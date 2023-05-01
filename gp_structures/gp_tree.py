import numpy as np
import gp_patch.deap_fix as deap_fix
from tools.tree_runner import run_tree
from deap import gp
from gp_terminals.gp_image import GPImage


def sigmoid_activation(x):
    return 1/(1 + np.exp(-x))


class GPTree:
    def __init__(self, pset, min_depth: int = 2, max_depth: int = 10, tree: gp.PrimitiveTree = None) -> None:
        self.pset = pset
        if tree is None:
            expr = deap_fix.genFull(pset, min_=min_depth, max_=max_depth)
            self.tree = gp.PrimitiveTree(expr)
        else:
            self.tree = tree
        
    def predict(self, image: GPImage) -> float:
        x = run_tree(image, self.tree)
        return sigmoid_activation(x)