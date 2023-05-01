import random
import gp_patch.deap_fix as deap_fix

from tqdm import tqdm
from copy import deepcopy
from deap import gp, base

from gp_structures.gp_dataset import GPDataset
from gp_structures.gp_tree import GPTree
from gp_utils.gp_saver import save_gp_tree
from tools.pset_generator import generate_pset
from sklearn.metrics import accuracy_score

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

    def fit(self, dataset: GPDataset) -> None:
        """
        Fit training dataset to classifier.

        Parameter
        ---------
        dataset: GPDataset
            Dataset that will be used by classifier.
        """
        
        # Generate populatiom
        self.population = [
            GPTree(self.pset, self.min_tree_depth, self.max_tree_depth)
            for _ in range(self.population_size)
        ]

        # Store dataset
        self.dataset = dataset

        # Start genetic loop
        bar = tqdm(range(self.generations))
        for _ in bar:
            self._evolve()
            bar.set_postfix({"best:": self._fitness(self.get_best(), dataset)})

    def _evolve(self) -> None:
        """
        Create new generation by crossover/mutation and add it to population.
        """

        # Loop through each individual
        for i in range(self.population_size):
            individual = self.population[i]
            # Mutation
            if random.uniform(0, 1) < self.mutation_rate:
                self.population += self._mutation(individual)
            
            # Crossover
            if random.uniform(0, 1) < self.crossover_rate:
                ind1 = random.randint(0, self.population_size-1)
                ind2 = random.randint(0, self.population_size-1)
                while ind1 == ind2:
                    ind1 = random.randint(0, self.population_size-1)
                    ind2 = random.randint(0, self.population_size-1)
                
                self.population += self._crossover(self.population[ind1], self.population[ind2])
        
        # Perform selection
        self._selection()
        # Save current best tree
        save_gp_tree(self.get_best())

    def _mutation(self, individual: GPTree) -> list[GPTree]:
        """
        Perform mutation.
        """
        
        mutated_individuals = [
            #GPTree(self.pset, tree=gp.mutEphemeral(deepcopy(individual.tree), "one")[0]),
            #GPTree(self.pset, tree=gp.mutNodeReplacement(deepcopy(individual.tree), self.pset)[0]),
            #GPTree(self.pset, tree=gp.mutInsert(deepcopy(individual.tree), self.pset)[0]),
            #GPTree(self.pset, tree=gp.mutShrink(deepcopy(individual.tree))[0]),
            GPTree(self.pset, tree=gp.mutUniform(deepcopy(individual.tree), self.toolbox.expr, self.pset)[0])
        ]
        
        return mutated_individuals

    def _crossover(self, parent1: GPTree, parent2: GPTree) -> list[GPTree]:
        """
        Perform crossover.
        """
        
        children = []
        children += list(gp.cxOnePoint(deepcopy(parent1.tree), deepcopy(parent2.tree)))
        children += list(gp.cxOnePointLeafBiased(deepcopy(parent1.tree), deepcopy(parent2.tree), self.crossover_rate))
        return [GPTree(self.pset, tree=tree) for tree in children]

    def _selection(self) -> None:
        """
        Selection of best individuals and removing the worst ones.
        """

        fitness_values = [(x, self._fitness(x)) for x in self.population]
        fitness_values.sort(key= lambda t: -t[1])
        fitness_values = fitness_values[:self.population_size]
        self.population = [t[0] for t in fitness_values]


    def _fitness(self, individual: GPTree) -> float:
        """
        Calculate fitness value of individual based on accuracy.

        Parameter
        ---------
        individual: GPTree
            Individual for fitness calculation.

        Returns
        -------
        float: The fitness value.
        """
        return self.evaluate(individual, self.dataset, accuracy_score)

    def evaluate(self, gptree: GPTree, dataset: GPDataset, metric) -> float:
        """
        Evaluate specific tree on dataset by specified metric.
        """

        predictions = []
        true_values = []
        for x in dataset:
            pred = gptree.predict(x[0])
            predictions.append(dataset.classes[1] if pred > 0.5 else dataset.classes[0])
            true_values.append(x[1])
        
        return metric(true_values, predictions)

    def get_best(self) -> GPTree:
        """
        Return best individual.

        Returns
        -------
        GPTree: Tree with highest fitness value.
        """

        return self.population[0]
