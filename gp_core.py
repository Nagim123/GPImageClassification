

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
    
    def fitness():
        pass

    def selection():
        pass

    def evolve():
        pass