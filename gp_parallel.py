from multiprocessing import Process, Queue

def calculate_fitness_pairs(fitness_fn, part: list, result_queue: Queue):
    fitness_values = [(x, fitness_fn(x)) for x in part]
    result_queue.put(fitness_values)

def parallel_fitness(fitness_fn, population: list, n_processes: int) -> list[tuple[any, float]]:
    if n_processes == 1:
        return [(x, fitness_fn(x)) for x in population]

    part_size = len(population) // n_processes
    parts = []

    # Loop over the number of parts
    for i in range(n_processes):
        start = i * part_size
        end = start + part_size

        if i == n_processes - 1:
            end = len(population)

        # Add the part to the list of parts
        parts.append(population[start:end])

    result_queue = Queue()
    processes = []
    for part in parts:
        p = Process(target=calculate_fitness_pairs, args=(fitness_fn, 2, result_queue,))
        p.start()
        processes.append(p)

    fitness_values = []
    for i in range(n_processes):
        # Get the result from the queue
        fitness_values += result_queue.get()

    for process in processes:
        process.join()

    return fitness_values