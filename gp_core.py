import numpy as np

import deap_fix
import gp_operators as ops
import copy

from tqdm import tqdm
from gp_dataset import GPDataset
from gp_tree import GPTree
import random
from deap import gp
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_percentage import GPPercentage
from gp_terminals.gp_percentage_size import GPPercentageSize
from typing import List

from gp_operators import agg_max, agg_mean, agg_min, agg_stdev, pool, add, conv, mul, div, sub


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
                 crossover_rate: float = 0.5,
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
        self.pset = pset
        self.population: List[GPTree] = []

    def _fitness(self, individual: GPTree, dataset: GPDataset) -> float:
        """
        Calculation of fitness based on accuracy.
        """

        correct = 0
        for i in range(len(dataset)):
            pred = individual.predict(dataset[i][0])
            if pred > 0.5:
                pred = dataset.classes[1]
            else:
                pred = dataset.classes[0]
            correct += pred == dataset[i][1]
        return correct / len(dataset)

    def _selection(self, dataset: GPDataset) -> None:
        """
        Selection of best individuals and removing the worst ones.
        """

        self.population.sort(key=lambda x: -self._fitness(x, dataset))
        self.population = self.population[:self.population_size]

    def _evolve(self, dataset: GPDataset) -> None:
        """
        Create new generation by crossover/mutation and add it to population.
        """

        print('evolve')
        for _ in range(round(self.crossover_rate * self.population_size)):
            r1, r2 = random.randint(0, self.population_size - 1), random.randint(0, self.population_size - 1)
            children = gp.cxOnePoint(copy.deepcopy(self.population[r1].tree), copy.deepcopy(self.population[r2].tree))
            self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]
            # children = gp.cxOnePointLeafBiased(copy.deepcopy(self.population[r1].tree),
            #                                    copy.deepcopy(self.population[r2].tree), self.crossover_rate)
            # self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]
        print('cross finished')
        for i in range(self.population_size):
            if random.uniform(0, 1) > self.mutation_rate:
                expr = None
                while expr is None:
                    try:
                        expr = deap_fix.genFull(self.pset, self.min_tree_depth, self.max_tree_depth)
                    except:
                        pass

                self.population += [
                    # GPTree(self.pset, tree=gp.mutEphemeral(copy.deepcopy(self.population[i].tree), "one")[0]),
                    # GPTree(self.pset, tree=gp.mutNodeReplacement(copy.deepcopy(self.population[i].tree), self.pset)[0]),
                    # GPTree(self.pset, tree=gp.mutInsert(copy.deepcopy(self.population[i].tree), self.pset)[0]),
                    # GPTree(self.pset, tree=gp.mutShrink(copy.deepcopy(self.population[i].tree))[0]),
                    GPTree(self.pset, tree=gp.mutUniform(copy.deepcopy(self.population[i].tree), expr, self.pset)[0])
                ]
        print('mutation finished')
        self._selection(dataset)
        print('selection finished')
        print()

    def fit(self, dataset) -> None:
        """
        Fit training dataset to classifier.
        """

        for _ in range(self.population_size):
            gptree = None
            while gptree == None:
                try:
                    gptree = GPTree(self.pset, self.min_tree_depth, self.max_tree_depth)
                except:
                    pass
            self.population.append(gptree)
            print(str(gptree.tree))
            print()
            print('*'*10)
            print()

        bar = tqdm(range(self.generations))
        for gen in bar:
            self._evolve(dataset)
            bar.set_postfix({"best:": self._fitness(self.get_best(), dataset)})

    def get_best(self) -> GPTree:
        """
        Return best individual.
        """

        return self.population[0]

    def predict(self, dataset) -> List[float]:
        """
        Predict data on specific dataset.
        """

        if (len(self.population) == 0):
            raise Exception("Call fit() before predict()!")

        the_best = self.get_best()
        result = []

        for i in range(len(dataset)):
            result.append(the_best.predict(dataset[i][0]))
        return result
