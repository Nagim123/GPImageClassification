import random

import deap.gp

from gp_dataset import GPDataset
from gp_tree import GPTree

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

        self.population = [GPTree() for _ in range(population_size)]
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
        for _ in range(self.crossover_rate * self.population_size // 2):
            r1, r2 = random.randint(self.population_size), random.randint(self.population_size)
            children = deap.gp.cxOnePoint(self.population[r1].tree, self.population[r2].tree)
            self.population += [GPTree(children[0]), GPTree(children[1])]
        for i in range(self.population_size):
            if random.uniform(0, 1) > self.mutation_rate:
                self.population += GPTree(deap.gp.mutNodeReplacement())
        self.selection()