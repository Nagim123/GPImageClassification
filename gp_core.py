import numpy as np
import gp_operators as ops

from tqdm import tqdm
from gp_dataset import GPDataset
from gp_tree import GPTree
import random
from deap import gp
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_image import GPImage

from gp_operators import agg_max, agg_mean, agg_min, agg_stdev, pool, add, conv


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
        pset.addPrimitive(ops.agg_mean, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_stdev, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_max, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.agg_min, [GPImage, GPPoint, GPSize, GPCutshape], float)
        pset.addPrimitive(ops.conv, [GPImage, GPFilter], GPImage)
        pset.addPrimitive(ops.pool, [GPImage], GPImage)

        # Additional info
        shape_names = ["rec", "col", "row", "elp"]
        iw, ih = train_dataset.size[0], train_dataset.size[1]
        
        pset.context["Filter"] = GPFilter
        pset.context["Shape"] = GPCutshape
        pset.context["Point"] = GPPoint
        pset.context["Size"] = GPSize

        #Terminal set
        #Generate random kernel filter with values in [-3, 3]
        pset.addEphemeralConstant("Filter", lambda: GPFilter((np.random.rand(3,3)-0.5)*6), GPFilter)
        pset.addEphemeralConstant("Shape", lambda: GPCutshape(shape_names[np.random.randint(0, len(shape_names))]), GPCutshape)
        
        pset.addEphemeralConstant("Point", lambda: GPPoint(
            int(iw*np.random.uniform(low=0.05, high=0.9)),
            int(ih*np.random.uniform(low=0.05, high=0.9))
        ),GPPoint)
        
        pset.addEphemeralConstant("Size", lambda: GPSize(
            int(iw*np.random.uniform(low=0.15, high=0.75)),
            int(ih*np.random.uniform(low=0.15, high=0.75)),
        ), GPSize)
        self.pset = pset
        self.population = [GPTree(pset, min_tree_depth, max_tree_depth) for _ in range(population_size)]
        
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
        self.population.sort(key=lambda x: self.fitness(x))
        self.population = self.population[:self.population_size]

    def evolve(self):
        for _ in range(round(self.crossover_rate * self.population_size)):
            r1, r2 = random.randint(0, self.population_size - 1), random.randint(0, self.population_size - 1)
            children = gp.cxOnePoint(self.population[r1].tree, self.population[r2].tree)
            self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]
        for i in range(self.population_size):
            if random.uniform(0, 1) > self.mutation_rate:
                self.population += [GPTree(self.pset, tree=gp.mutNodeReplacement(self.population[i].tree, self.pset))]
                # TODO: different mutations
        self.selection()
    
    def fit(self, dataset):
        bar = tqdm(range(self.generations))
        for gen in bar:
            self.evolve()


    def get_best(self):
        return self.population[-1]


train_dataset = GPDataset("toydataset/train", (20, 20))
dfn = "lambda ARG0: add(agg_stdev(pool(pool(conv(conv(conv(ARG0, GPFilter(np.array([[1.4750853742645986, 2.217925634734678, -1.452910671550328], [-1.0377072159661902, 1.2274511414191416, -2.17232973222214], [-1.8562016847073597, 1.8131083190287267, -0.03975583387378934]]))), GPFilter(np.array([[-0.8498909470568146, 2.5557504679757557, 2.3213637326538272], [-2.5262083014087757, -0.21983553936789568, -0.06769948216963861], [1.3529076995083484, -1.733522192162067, 0.25437282208981093]]))), GPFilter(np.array([[1.0166347285473951, -0.10742190195121215, 1.230975515028125], [2.077937361531241, 2.389617881048908, -1.2698677433898524], [2.178971457663086, -2.08603465358833, -1.5898664533470726]]))))), GPPoint(12,5), GPSize(14,11), GPCutshape('col')), agg_mean(conv(pool(conv(pool(conv(ARG0, GPFilter(np.array([[2.814737820668662, -2.6284082756506333, -2.353871193777574], [-1.7867007178568683, -2.9642651390247465, -0.019793276587760866], [1.424499082208433, -0.9801296998375639, -1.7936892347200795]])))), GPFilter(np.array([[-1.455701909056084, 0.5148049682042366, 1.0435373057577892], [-2.361942980840026, -1.270867050887598, -1.743855493800901], [-0.6762807003118405, -1.7796328306190998, 0.15413521365338756]])))), GPFilter(np.array([[0.15897893279603648, 1.23230098597239, 1.7208866185363358], [-2.277286600279978, 1.3635288949656446, 2.774295536090584], [1.539791372897039, 2.255695406873995, -0.19855259137973325]]))), GPPoint(16,5), GPSize(5,7), GPCutshape('elp')))"
dfn = eval(dfn)
dfn(train_dataset[0][0])