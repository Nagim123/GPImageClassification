import numpy as np
import gp_operators as ops
from gp_dataset import GPDataset
from gp_tree import GPTree
import random
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
                 train_dataset: GPDataset,
                 test_dataset: GPDataset,
                 classes: list,
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
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        # Tree definition.
        pset = gp.PrimitiveSetTyped([GPImage], float)
        # Function set
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

        # Additional info
        shape_names = ["rect", "col", "row", "eps"]
        iw, ih = train_dataset.size[0], train_dataset.size[1]

        # Terminal set
        # Generate random kernel filter with values in [-3, 3]
        pset.addEphemeralConstant("Filter", lambda: GPFilter((np.random.rand(3, 3) - 0.5) * 6))
        pset.addEphemeralConstant("Shape", lambda: GPCutshape(shape_names[np.random.randint(0, len(shape_names))]))

        pset.addEphemeralConstant("Point", lambda: GPPoint(
            iw * np.random.uniform(low=0.05, high=0.9), ih * np.random.uniform(low=0.05, high=0.9)
        ))

        pset.addEphemeralConstant("Size", lambda: GPSize(
            iw * np.random.uniform(low=0.15, high=0.75), ih * np.random.uniform(low=0.15, high=0.75)
        ))

        self.pset = pset

        self.population = [GPTree(min_tree_depth, max_tree_depth, pset) for _ in range(population_size)]
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes

    def fitness(self, individual: GPTree):
        correct = 0
        for i in range(len(self.train_dataset)):
            pred = individual.feed(self.train_dataset[i][0])
            if pred > 0.5:
                pred = self.classes[1]
            else:
                pred = self.classes[0]
            correct += pred == self.train_dataset[i][1]
        return correct / len(self.train_dataset)

    def selection(self):
        self.population.sort(key=lambda x: self.fitness(x))
        self.population = self.population[:self.population_size]

    def evolve(self):
        for _ in range(round(self.crossover_rate * self.population_size)):
            r1, r2 = random.randint(0, self.population_size - 1), random.randint(0, self.population_size - 1)
            children = gp.cxOnePoint(self.population[r1].tree, self.population[r2].tree)
            self.population += [GPTree(children[0], self.pset), GPTree(children[1], self.pset)]
        for i in range(self.population_size):
            if random.uniform(0, 1) > self.mutation_rate:
                self.population += GPTree(gp.mutNodeReplacement(self.population[i].tree, self.pset), self.pset)
                # TODO: different mutations
        self.selection()
