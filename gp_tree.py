import numpy as np
import gp_operators as ops

from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage

class GPTree:
    def __init__(self, image_size: GPSize, min_depth: int, max_depth: int) -> None:
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

        #Additional info
        shape_names = ["rect", "col", "row", "eps"]
        iw, ih = image_size.w, image_size.h

        #Terminal set
        #Generate random kernel filter with values in [-3, 3]
        pset.addEphemeralConstant("Filter", lambda: GPFilter((np.random.rand(3,3)-0.5)*6))
        pset.addEphemeralConstant("Shape", lambda: GPCutshape(shape_names[np.random.randint(0, len(shape_names))]))
        
        pset.addEphemeralConstant("Point", lambda: GPPoint(
            iw*np.random.uniform(low=0.05, high=0.9), ih*np.random.uniform(low=0.05, high=0.9)
        ))
        
        pset.addEphemeralConstant("Size", lambda: GPSize(
            iw*np.random.uniform(low=0.15, high=0.75), ih*np.random.uniform(low=0.15, high=0.75)
        ))
        self.pset = pset
        self.expr = gp.genFull(self.pset, min_=min_depth, max_=max_depth)
        self.tree = gp.PrimitiveTree(self.expr)
    
    def feed(self, image: GPImage) -> float:
        return gp.compile(self.expr, self.pset)(image)