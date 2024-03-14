import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot
def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)

# Interpolation crossover
def crossover(f, parents: List[Entity]) -> Entity:
    a = random.uniform(0, 1)  # Losowa wartość a z przedziału [0, 1]
    p1 = random.choice(parents)
    p2 = random.choice(parents)

    x = [0.0] * len(p1.x)
    s = [0.0] * len(p1.x)
    for i in range(len(p1.x)):
        x[i] = a * p1.x[i] + (1 - a) * p2.x[i]  # Interpolacja wartości współrzędnych
        s[i] = a * p1.s[i] + (1 - a) * p2.s[i]  # Interpolacja wartości sigma
    return Entity(x, s)
