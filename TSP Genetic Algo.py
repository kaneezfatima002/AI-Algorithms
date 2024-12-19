import random

def create_distance_matrix(num_cities):
    distance_matrix = [[random.randint(10, 100) for _ in range(num_cities)] for _ in range(num_cities)]
    for i in range(num_cities):
        distance_matrix[i][i] = 0
    return distance_matrix

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        tour = random.sample(range(num_cities), num_cities)
        population.append(tour)
    return population


def calculate_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i]][tour[i + 1]]
    distance += distance_matrix[tour[-1]][tour[0]]
    return distance


def evaluate_fitness(population, distance_matrix):
    fitness = []
    for tour in population:
        distance = calculate_distance(tour, distance_matrix)
        fitness.append(1 / distance)
    return fitness


def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parent1 = random.choices(population, weights=probabilities, k=1)[0]
    parent2 = random.choices(population, weights=probabilities, k=1)[0]
    return parent1, parent2

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1
    return child

def mutate(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations):
    population = initialize_population(pop_size, len(distance_matrix))
    for _ in range(num_generations):
        fitness = evaluate_fitness(population, distance_matrix)
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    best_tour = min(population, key=lambda tour: calculate_distance(tour, distance_matrix))
    return best_tour, calculate_distance(best_tour, distance_matrix)

num_cities = 5
distance_matrix = create_distance_matrix(num_cities)
best_tour, best_distance = genetic_algorithm_tsp(distance_matrix, pop_size=10, num_generations=100)
print("Best Tour:", best_tour)
print("Best Distance:", best_distance)