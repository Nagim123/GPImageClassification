import numpy as np
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

        #Terminal set
        #Generate random kernel filter with values in [-3, 3]
        pset.addEphemeralConstant("Filter", lambda: GPFilter((np.random.rand(3,3)-0.5)*6), GPFilter)
        pset.addEphemeralConstant("Shape", lambda: GPCutshape(shape_names[np.random.randint(0, len(shape_names))]), GPCutshape)
        
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
        #self.population = [GPTree(pset, min_tree_depth, max_tree_depth) for _ in range(population_size)]
        for _ in range(population_size):
            print(f"Created {_}/{population_size}")
            gptree = None
            while gptree == None:
                try:
                    gptree = GPTree(pset, min_tree_depth, max_tree_depth)
                except:
                    pass
            
            self.population.append(gptree)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes
        print("DONE CREATION")

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
        self.population.sort(key=lambda x: -self.fitness(x))
        self.population = self.population[:self.population_size]

    def evolve(self):
        for _ in range(round(self.crossover_rate * self.population_size)):
            r1, r2 = random.randint(0, self.population_size - 1), random.randint(0, self.population_size - 1)
            children = gp.cxOnePoint(copy.deepcopy(self.population[r1].tree), copy.deepcopy(self.population[r2].tree))
            self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]
        for i in range(self.population_size):
            if random.uniform(0, 1) > self.mutation_rate:
                self.population += [GPTree(self.pset, tree=gp.mutEphemeral(copy.deepcopy(self.population[i].tree), "one")[0])]
                # TODO: different mutations
        self.selection()
    
    def fit(self, dataset):
        bar = tqdm(range(self.generations))
        for gen in bar:
            self.evolve()
            bar.set_postfix({"best:":self.fitness(self.get_best())})
            print(str(self.get_best().tree))

    def get_best(self) -> GPTree:
        return self.population[0]


# train_dataset = GPDataset("toydataset/train", (20, 20))
# dfn = "lambda ARG0: agg_min(conv(pool(ARG0), GPFilter(np.array([[-1.9312678938815866, -2.301198424586718, 2.4333607536213955], [-1.5678849281981493, 2.3056104677348155, 2.10001006247031], [-0.5096000332508739, 0.45372257013040196, 0.8309849412175507]]))), GPPercentage(0.6539595839796143,0.24898311846783155), GPPercentageSize(0.3061538354389799,0.5865971625482215), GPCutshape('elp'))"
# dfn = eval(dfn)
# print(dfn(train_dataset[0][0]))