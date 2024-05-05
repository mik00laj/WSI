import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from typing import Tuple

MU = 128
LAMBDA = 512
SIGMA = 0.1
ITERATIONS = 200

def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)

def evolution_es_min(f, MU: int, LAMBDA: int, SIGMA: float, ITERATIONS: int) -> Tuple[np.ndarray, float, list]:
    population = list()
    x_history = []
    y_history = []
    f_history = []

    for i in range(MU):
        x = np.random.uniform(-20.0, 20.0)
        y = np.random.uniform(-20.0, 20.0)
        population.append(([x, y], np.array([SIGMA] * 2)))

    score = [f(x, y) for [x, y], _ in population]
    best_ever = min(zip(population, score), key=lambda a: a[1])

    min_ypoints = [0.0] * ITERATIONS
    for epoch in range(ITERATIONS):
        parents = [tournament_selection_min(population) for _ in range(LAMBDA)]
        children = [crossover(f, parents) for _ in range(LAMBDA)]
        children = [mutation(f, e) for e in children]

        best_epoch = min(zip(children, score), key=lambda a: a[1])
        if best_epoch[1] < best_ever[1]:
            best_ever = copy.deepcopy(best_epoch)

        population += children
        score = [f(x, y) for [x, y], _ in population]

        population = [indiv for _, indiv in sorted(zip(score, population), key=lambda pair: pair[0])]
        population = population[:MU]

        min_ypoints[epoch] = f(*population[0][0])
        f_history.append(min_ypoints[epoch])
        x_history.append(population[0][0][0])
        y_history.append(population[0][0][1])
        print("Pokolenie: ", epoch, "f(x,y)= ", min_ypoints[epoch])

    return best_ever[0][0], f(*best_ever[0][0]), population, f_history, x_history, y_history

def evolution_es_max(f, MU: int, LAMBDA: int, SIGMA: float, ITERATIONS: int) -> Tuple[np.ndarray, float, list]:
    population = list()
    x_history = []
    y_history = []
    f_history = []
    # Tworzenie polpulacji początkowej
    for i in range(MU):
        x = np.random.uniform(-20.0, 20.0)
        y = np.random.uniform(-20.0, 20.0)
        population.append(([x, y], np.array([SIGMA] * 2)))

    score = [f(x, y) for [x, y], _ in population]
    best_ever = max(zip(population, score), key=lambda a: a[1])

    min_ypoints = [0.0] * ITERATIONS
    for epoch in range(ITERATIONS):
        parents = [tournament_selection_max(population) for _ in range(LAMBDA)]
        children = [crossover(f, parents) for _ in range(LAMBDA)]
        children = [mutation(f, e) for e in children]

        best_epoch = max(zip(children, score), key=lambda a: a[1])
        if best_epoch[1] > best_ever[1]:
            best_ever = copy.deepcopy(best_epoch)

        population += children
        score = [f(x, y) for [x, y], _ in population]

        population = [indiv for _, indiv in sorted(zip(score, population), key=lambda pair: pair[0], reverse=True)]  # Sortowanie malejąco
        population = population[:MU]

        min_ypoints[epoch] = f(*population[0][0])
        f_history.append(min_ypoints[epoch])
        x_history.append(population[0][0][0])
        y_history.append(population[0][0][1])
        print("Pokolenie: ", epoch, "f(x,y)= ", min_ypoints[epoch])

    return best_ever[0][0], f(*best_ever[0][0]), population, f_history, x_history, y_history

def tournament_selection_max(population: list) -> Tuple[np.ndarray, np.ndarray]:
    tournament_size = 2
    tournament_contestants = random.choices(population, k=tournament_size)
    return max(tournament_contestants, key=lambda x: f(*x[0]))
def tournament_selection_min(population: list) -> Tuple[np.ndarray, np.ndarray]:
    tournament_size = 2
    tournament_contestants = random.choices(population, k=tournament_size)
    return min(tournament_contestants, key=lambda x: f(*x[0]))

#Krzyżowanie uśredniające w wariancie roszerzonym
def crossover(f, parents: list):
    a = random.uniform(0, 1)
    p1 = random.choice(parents)  # Losowy wybór pierwszego rodzica
    p2 = random.choice(parents)  # Losowy wybór drugiego rodzica

    p1_x, p1_sigma = p1
    p2_x, p2_sigma = p2

    crossover_child_x = [a * p1_x[i] + (1 - a) * p2_x[i] for i in range(len(p1_x))]
    crossover_child_sigma = [a * p1_sigma[i] + (1 - a) * p2_sigma[i] for i in range(len(p1_sigma))]

    return [crossover_child_x, crossover_child_sigma]

# Mutacja Gaussowska
def mutation(f, parent: Tuple[np.ndarray, np.ndarray]):
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

# FIND MINIMUM
# best_solution, best_fitness, population, f_history, x_history, y_history = evolution_es_min(f, MU, LAMBDA, SIGMA, ITERATIONS)
# FIND MAXIMUM
best_solution, best_fitness, population, f_history, x_history, y_history = evolution_es_max(f, MU, LAMBDA, SIGMA, ITERATIONS)

print("Best solution:", best_solution)
print("Objective function value at the best solution:", best_fitness)


# Wykres jakości punktów populacji
xpoints = list(range(ITERATIONS))
plt.scatter(xpoints, f_history, color='red', label='f_history')
plt.xlabel('Iterations')
plt.ylabel('Value of Function')
plt.title('Evolution Alghoritm')
plt.show()

# Create a range of x and y values for plotting
x_vals = np.linspace(-3.0, 3, 100)
y_vals = np.linspace(-3.0, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)
# Plot Contour Plot
plt.figure(figsize=(12, 10))
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.scatter(x_history, y_history, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Evolution Alghoritm')
plt.colorbar(contour)
plt.show()