import copy
import random
import gp_patch.deap_fix as deap_fix

from tqdm import tqdm
from deap import gp, base
from typing import List

from gp_structures.gp_dataset import GPDataset
from gp_structures.gp_tree import GPTree
from gp_utils.gp_saver import save_gp_tree
from tools.pset_generator import generate_pset

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

        
        self.pset = generate_pset()
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", deap_fix.genFull, pset=self.pset, min_=1, max_=3)
        self.population: List[GPTree] = []

    def _fitness(self, individual: GPTree, dataset: GPDataset) -> float:
        """
        Calculation of fitness based on accuracy.
        """

        correct = 0
        for i in range(len(dataset)):
            pred = individual.predict(dataset[i][0])
            pred = dataset.classes[1] if pred > 0.5 else dataset.classes[0]
            correct += pred == dataset[i][1]
        return correct / len(dataset)

    def _selection(self, dataset: GPDataset) -> None:
        """
        Selection of best individuals and removing the worst ones.
        """
        fitness_values = [(x, self._fitness(x, dataset)) for x in self.population]
        fitness_values.sort(key= lambda t: -t[1])
        fitness_values = fitness_values[:self.population_size]
        self.population = [t[0] for t in fitness_values]

    def _evolve(self, dataset: GPDataset) -> None:
        """
        Create new generation by crossover/mutation and add it to population.
        """

        # Perform crossover
        for _ in range(round(self.crossover_rate * self.population_size)):
            r1, r2 = random.randint(0, self.population_size - 1), random.randint(0, self.population_size - 1)
            children = gp.cxOnePoint(copy.deepcopy(self.population[r1].tree), copy.deepcopy(self.population[r2].tree))
            self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]
            children = gp.cxOnePointLeafBiased(copy.deepcopy(self.population[r1].tree),
                                                copy.deepcopy(self.population[r2].tree), self.crossover_rate)
            self.population += [GPTree(self.pset, tree=children[0]), GPTree(self.pset, tree=children[1])]

        # Perform mutation
        for i in range(self.population_size):
            if random.uniform(0, 1) < self.mutation_rate:
                self.population += [
                    #GPTree(self.pset, tree=gp.mutEphemeral(copy.deepcopy(self.population[i].tree), "one")[0]),
                    #GPTree(self.pset, tree=gp.mutNodeReplacement(copy.deepcopy(self.population[i].tree), self.pset)[0]),
                    #GPTree(self.pset, tree=gp.mutInsert(copy.deepcopy(self.population[i].tree), self.pset)[0]),
                    #GPTree(self.pset, tree=gp.mutShrink(copy.deepcopy(self.population[i].tree))[0]),
                    GPTree(self.pset, tree=gp.mutUniform(copy.deepcopy(self.population[i].tree), self.toolbox.expr, self.pset)[0])
                ]
        
        # Perform selection
        self._selection(dataset)
        # Save current best tree
        save_gp_tree(self.get_best())

    def fit(self, dataset: GPDataset) -> None:
        """
        Fit training dataset to classifier.
        """

        # Generate populatiom
        self.population = [
            GPTree(self.pset, self.min_tree_depth, self.max_tree_depth)
            for _ in range(self.population_size)
        ]

        # Start genetic loop
        bar = tqdm(range(self.generations))
        for _ in bar:
            self._evolve(dataset)
            bar.set_postfix({"best:": self._fitness(self.get_best(), dataset)})

    def get_best(self) -> GPTree:
        """
        Return best individual.
        """

        return self.population[0]

    def predict(self, dataset: GPDataset) -> List[float]:
        """
        Predict data on specific dataset.
        """

        if (len(self.population) == 0):
            raise RuntimeError("Call fit() before predict()!")

        the_best = self.get_best()
        return [the_best.predict(dataset[i][0]) for i in range(len(dataset))]
