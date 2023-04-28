import numpy as np
import gp_operators as ops

from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage

class GPImageClassifier:
    """
    Main class that implements Genetic Programming.
    """
    
    def __init__(self,
                population_size: int = 50,
                generations: int = 50,
                min_tree_depth: int = 2,
                max_tree_depth: int = 10,
                tournament_size: int = 7,
                mutation_rate: float = 0.2,
                crossover_rate: float = 0.8,
                elitism: int = 10
                ) -> None:
        """
        Initialize Genetic Programming algorithm.

        Parameter
        ---------
        population_size: int
            The number of individuals saved in memory.
        
        generations: int
            How much generations will be produced before termination.
        
        min_tree_size: int
            Minimum depth of generated trees.

        max_tree_size: int
            Maximum depth of generated trees.

        tournament_size: int
            ???.    
        """
        
        self.population_size = population_size
        self.generations = generations
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.tournamnt_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

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
        iw, ih = image_size.w, image_size.h #SIZES

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
        
    def fitness():
        pass

    def selection():
        pass

    def evolve():
        pass