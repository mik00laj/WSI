from dataclasses import dataclass
import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt


@dataclass
class Entity:
    x: np.ndarray
    s: np.ndarray


def evolution_es(f, x0: np.ndarray, mu: int, lambda_: int, s: float, stop: int):
    pop = list()
    if x0 is None:
        for i in range(mu):
            x0 = np.random.uniform(-100.0, 100.0, size=2)  # Modyfikacja dla dwóch wymiarów
            pop.append(Entity(x0, np.full_like(x0, s)))
    else:
        pop = [Entity(x0, np.full_like(x0, s))] * mu

    score = [f(*e.x) for e in pop]
    best_ever = min(zip(pop, score), key=lambda a: a[1])

    min_ypoints = [0.0] * stop
    for epoch in range(stop):
        children = [crossover(f, pop) for _ in range(lambda_)]
        children = [mutation(f, e) for e in children]

        best_epoch = min(zip(children, score), key=lambda a: a[1])
        if best_epoch[1] < best_ever[1]:
            best_ever = copy.deepcopy(best_epoch)

        pop += children
        score = [f(*e.x) for e in pop]
        pop = [x for _, x in sorted(zip(score, pop), key=lambda pair: pair[0])]
        pop = pop[:mu]

        min_ypoints[epoch] = f(*pop[0].x)
        print(f"{epoch}: {min_ypoints[epoch]}") 
    xpoints = list(range(stop))
    plt.plot(xpoints, min_ypoints, label="mu + lambda")
    plt.legend()
    return [best_ever[0].x, f(*best_ever[0].x)]


def crossover(f, parents: np.ndarray[Entity]) -> Entity:
    a = random.uniform(0, 1)
    p1 = random.choice(parents)
    p2 = random.choice(parents)

    x = a * p1.x + (1 - a) * p2.x
    s = a * p1.s + (1 - a) * p2.s
    return Entity(x, s)


def mutation(f, parent: Entity) -> Entity:
    n = len(parent.x)
    tau = 1 / math.sqrt(2.0 * n)
    taup = 1 / math.sqrt(2.0 * math.sqrt(n))

    a = random.normalvariate(0.0, 1.0)
    b = np.random.normal(0.0, 1.0, size=n)

    s = parent.s * np.exp(taup * a + tau * b)
    x = parent.x + s * np.random.normal(0.0, 1.0, size=n)
    return Entity(x, s)


def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


best_solution, best_fitness = evolution_es(f, None, mu=10, lambda_=30, s=0.1, stop=100)

print("Najlepsze rozwiązanie:", best_solution)
print("Wartość funkcji celu w najlepszym rozwiązaniu:", best_fitness)
