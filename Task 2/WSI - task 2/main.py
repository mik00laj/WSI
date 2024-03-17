import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from typing import Tuple

def evolution_es(f, x0: np.ndarray, MU: int, LAMBDA: int, SIGMA: float, ITERATIONS: int) -> Tuple[np.ndarray, float]:
    population = list()
    if x0 is None:
        for i in range(MU):
            x0 = np.random.uniform(-100.0, 100.0, size=2)  # Initializing population with random individuals
            population.append((x0, np.array([SIGMA] * len(x0))))
    else:
        population = [(x0, np.array([SIGMA] * len(x0)))] * MU

    score = [f(*x) for x, _ in population]  # Calculating the objective function value for each individual
    best_ever = min(zip(population, score), key=lambda a: a[1])  # Finding the best individual in the populationulation

    min_ypoints = [0.0] * ITERATIONS
    for epoch in range(ITERATIONS):
        # Parent selection: Tournament reproduction
        parents = [tournament_selection(population) for _ in range(LAMBDA)]

        # Crossover and MUtation
        children = [crossover(f, parents) for _ in range(LAMBDA)]
        children = [MUtation(f, e) for e in children]

        # Selecting the best individuals from the populationulation and offspring
        best_epoch = min(zip(children, score), key=lambda a: a[1])
        if best_epoch[1] < best_ever[1]:
            best_ever = copy.deepcopy(best_epoch)

        # Combining populationulation and offspring
        population += children

        # Calculating the objective function value for the new populationulation
        score = [f(*x) for x, _ in population]

        # Sorting the populationulation based on the objective function value and selecting the best individuals
        population = [x for _, x in sorted(zip(score, population), key=lambda pair: pair[0])]
        population = population[:MU]

        # Saving the objective function value of the best individual in the current generation
        min_ypoints[epoch] = f(*population[0])
        print(f"{epoch}: {min_ypoints[epoch]}")

    # Plotting the objective function value in successive generations
    xpoints = list(range(ITERATIONS))
    plt.plot(xpoints, min_ypoints, label="MU + lambda")
    plt.legend()

    # Returning the best found solution and its objective function value
    return best_ever[0][0], f(*best_ever[0][0])

# Tournament selection function
def tournament_selection(populationulation: list) -> Tuple[np.ndarray, np.ndarray]:
    tournament_size = 2  # Tournament size
    tournament_contestants = random.choices(populationulation, k=tournament_size)  # Random selection of tournament participants
    return min(tournament_contestants, key=lambda x: f(*x[0]))  # Returning the tournament winner

# Crossover function
def crossover(f, parents: list):
    a = random.uniform(0, 1)
    p1, p2 = random.sample(parents, 2)  # Selecting two random parents

    x1, SIGMA1 = p1
    x2, SIGMA2 = p2

    x = a * x1 + (1 - a) * x2  # Interpolating coordinate values
    SIGMA = a * SIGMA1 + (1 - a) * SIGMA2  # Interpolating SIGMA values
    return x, SIGMA


# MUtation function
def MUtation(f, parent: Tuple[np.ndarray, np.ndarray]):
    x, SIGMA = parent

    n = float(len(x))
    tau = 1 / np.sqrt(2.0 * n)
    taup = 1 / np.sqrt(2.0 * np.sqrt(n))

    a = random.normalvariate(0.0, 1.0)
    for i in range(len(x)):
        b = random.normalvariate(0.0, 1.0)
        SIGMA[i] = SIGMA[i] * np.exp(taup * a + tau * b)
        x[i] = x[i] + SIGMA[i] * random.normalvariate(0.0, 1.0)

    return x, SIGMA

# Objective function f(x, y)
def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)

# Running the ES algorithm
best_solution, best_fitness = evolution_es(f, None, MU=10, LAMBDA=100, SIGMA=0.1, ITERATIONS=1000)

print("Best solution:", best_solution)
print("Objective function value at the best solution:", best_fitness)
