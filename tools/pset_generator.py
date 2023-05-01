import numpy as np
import gp_operators as ops

from deap import gp, base
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_percentage import GPPercentage
from gp_terminals.gp_percentage_size import GPPercentageSize

def generate_pset() -> gp.PrimitiveSetTyped:
    
    # Tree definition.
    pset = gp.PrimitiveSetTyped("MainTree", [GPImage], float)
    # Function set
    pset.addPrimitive(ops.add, [float, float], float)
    pset.addPrimitive(ops.sub, [float, float], float)
    pset.addPrimitive(ops.mul, [float, float], float)
    pset.addPrimitive(ops.div, [float, float], float)
    pset.addPrimitive(ops.agg_mean, [GPImage, GPPercentage, GPPercentageSize, GPCutshape], float)
    pset.addPrimitive(ops.agg_stdev, [GPImage, GPPercentage, GPPercentageSize, GPCutshape], float)
    pset.addPrimitive(ops.agg_max, [GPImage, GPPercentage, GPPercentageSize, GPCutshape], float)
    pset.addPrimitive(ops.agg_min, [GPImage, GPPercentage, GPPercentageSize, GPCutshape], float)
    pset.addPrimitive(ops.conv, [GPImage, GPFilter], GPImage)
    pset.addPrimitive(ops.pool, [GPImage], GPImage)

    # Additional info
    shape_names = ["rec", "col", "row", "elp"]

    pset.context["Filter"] = GPFilter
    pset.context["Shape"] = GPCutshape
    pset.context["Point"] = GPPercentage
    pset.context["Size"] = GPPercentageSize
    pset.context["Constant"] = float

    # Terminal set
    # Generate random kernel filter with values in [-3, 3]
    pset.addEphemeralConstant("Filter", lambda: GPFilter(np.random.randint(-3, 3, size=(3, 3))), GPFilter)
    pset.addEphemeralConstant("Shape", lambda: GPCutshape(shape_names[np.random.randint(0, len(shape_names))]),
                                GPCutshape)

    pset.addEphemeralConstant("Constant", lambda: np.random.randint(-5, 5), float)

    pset.addEphemeralConstant("Point", lambda: GPPercentage(
        np.random.uniform(low=0.05, high=0.9),
        np.random.uniform(low=0.05, high=0.9)
    ), GPPercentage)

    pset.addEphemeralConstant("Size", lambda: GPPercentageSize(
        np.random.uniform(low=0.15, high=0.75),
        np.random.uniform(low=0.15, high=0.75),
    ), GPPercentageSize)

    return pset