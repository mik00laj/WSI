import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from typing import List
from dataclasses import dataclass

@dataclass
class Entity:
    x: List[float]
    s: List[float]

def evolution_es(f, x0: List[float], mu: int, lambda_: int, s: float, stop: int):
    pop = list()
    if x0 is None:
        for i in range(mu):
            x0 = np.random.uniform(-100.0, 100.0, size=2)  # Inicjalizacja populacji losowymi osobnikami
            pop.append(Entity(x0, [s] * len(x0)))
    else:
        pop = [Entity(x0, [s] * len(x0))] * mu

    score = list(map(lambda e: f(*e.x), pop))  # Obliczenie wartości funkcji celu dla każdego osobnika
    best_ever = min(zip(pop, score), key=lambda a: a[1])  # Znalezienie najlepszego osobnika w populacji

    min_ypoints = [0.0] * stop
    for epoch in range(stop):
        # Selekcja rodziców: Reprodukcja turniejowa
        parents = [tournament_selection(pop) for _ in range(lambda_)]

        # Krzyżowanie i mutacja
        children = [crossover(f, parents) for _ in range(lambda_)]
        children = list(map(lambda e: mutation(f, e), children))

        # Wybór najlepszych osobników spośród populacji i potomstwa
        best_epoch = min(zip(children, score), key=lambda a: a[1])
        if best_epoch[1] < best_ever[1]:
            best_ever = copy.deepcopy(best_epoch)

        # Połączenie populacji i potomstwa
        pop += children

        # Obliczenie wartości funkcji celu dla nowej populacji
        score = list(map(lambda e: f(*e.x), pop))

        # Sortowanie populacji na podstawie wartości funkcji celu i wybór najlepszych osobników
        pop = [x for _, x in sorted(zip(score, pop), key=lambda pair: pair[0])]
        pop = pop[:mu]

        # Zapisanie wartości funkcji celu najlepszego osobnika w danym pokoleniu
        min_ypoints[epoch] = f(*pop[0].x)
        print(f"{epoch}: {min_ypoints[epoch]}")

    # Wykres wartości funkcji celu w kolejnych pokoleniach
    xpoints = list(range(stop))
    plt.plot(xpoints, min_ypoints, label="mu + lambda")
    plt.legend()

    # Zwrócenie najlepszego znalezionego rozwiązania i jego wartości funkcji celu
    return [best_ever[0].x, f(*best_ever[0].x)]

# Funkcja selekcji turniejowej
def tournament_selection(population: List[Entity]) -> Entity:
    tournament_size = 2  # Rozmiar turnieju
    tournament_contestants = random.choices(population, k=tournament_size)  # Losowy wybór uczestników turnieju
    return min(tournament_contestants, key=lambda x: f(*x.x))  # Zwrócenie zwycięzcy turnieju

# Funkcja krzyżowania
def crossover(f, parents: List[Entity]) -> Entity:
    # Tutaj wykorzystywany jest losowy wybór rodziców do krzyżowania
    a = random.uniform(0, 1)
    p1 = random.choice(parents)
    p2 = random.choice(parents)

    x = [0.0] * len(p1.x)
    s = [0.0] * len(p1.x)
    for i in range(len(p1.x)):
        x[i] = a * p1.x[i] + (1 - a) * p2.x[i]  # Interpolacja wartości współrzędnych
        s[i] = a * p1.s[i] + (1 - a) * p2.s[i]  # Interpolacja wartości sigma
    return Entity(x, s)

# Funkcja mutacji
def mutation(f, parent: Entity) -> Entity:
    x = [0.0] * len(parent.x)
    s = [0.0] * len(parent.x)

    n = float(len(x))
    tau = 1 / np.sqrt(2.0 * n)
    taup = 1 / np.sqrt(2.0 * np.sqrt(n))

    a = random.normalvariate(0.0, 1.0)
    for i in range(len(parent.x)):
        b = random.normalvariate(0.0, 1.0)
        s[i] = parent.s[i] * np.exp(taup * a + tau * b)
        x[i] = parent.x[i] + s[i] * random.normalvariate(0.0, 1.0)

    return Entity(x, s)

# Funkcja celu f(x, y)
def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)

# Uruchomienie algorytmu ES
best_solution, best_fitness = evolution_es(f, None, mu=10, lambda_=30, s=0.1, stop=100)

print("Najlepsze rozwiązanie:", best_solution)
print("Wartość funkcji celu w najlepszym rozwiązaniu:", best_fitness)
