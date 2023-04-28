from deap import gp

class GPImageClassifier:
    """
    Main class that implements Genetic Programming.
    """
    
    def __init__(self,
                population_size = 50,
                generations = 50,
                min_tree_depth = 2,
                max_tree_depth = 10,
                tournament_size = 7,
                mutation_rate = 0.2,
                crossover_rate = 0.8,
                elitism = 10
                ) -> None:
        """
        Initialize Genetic Programming algorithm.
        """
        
        self.population_size = population_size
        self.generations = generations
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.tournamnt_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.pset = gp.PrimitiveSetTyped([GPImage, ])
    
    def fitness():
        pass

    def selection():
        pass

    def evolve():
        pass