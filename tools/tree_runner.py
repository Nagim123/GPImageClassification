import numpy as np

from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_percentage import GPPercentage
from gp_terminals.gp_percentage_size import GPPercentageSize

from gp_operators import agg_max, agg_mean, agg_min, agg_stdev, pool, add, conv, mul, div, sub
from deap import gp

def run_tree(input: GPImage, tree: gp.PrimitiveTree) :
    lambda_function = eval('lambda ARG0: ' + str(tree))
    return lambda_function(input)