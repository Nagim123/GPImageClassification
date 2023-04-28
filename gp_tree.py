import numpy as np
import gp_operators as ops

from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage

class GPTree:
    def __init__(self) -> None:
        # Tree definition.
        pset = gp.PrimitiveSetTyped([GPImage], float)
        #Function set
        pset.addPrimitive(ops.add, [float, float], float)
        pset.addPrimitive(ops.sub, [float, float], float)
        pset.addPrimitive(ops.mul, [float, float], float)
        pset.addPrimitive(ops.div, [float, float], float)
        pset.addPrimitive(ops.agg_mean, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_stdev, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_max, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_min, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.conv, [GPImage, GPFilter], GPImage)
        pset.addPrimitive(ops.pool, [GPImage], GPImage)

        #Terminal set
        pset.addEphemeralConstant("Filter", lambda: GPFilter(np.random.rand()))

        self.pset = pset


    def feed(self, image: GPImage) -> float:
        return 1.0