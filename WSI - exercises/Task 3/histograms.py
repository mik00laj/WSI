import random
import numpy as np
import matplotlib.pyplot as plt
from main import MinMaxAgent, GreedyAgent, NinjaAgent, RandomAgent, run_game


def test_agent(agent_type, depth, trials=1000):
    scores = []
    for _ in range(trials):
        vector = [random.randint(-10, 10) for _ in range(15)]
        minmax_agent = MinMaxAgent(max_depth=depth)
        second_agent = agent_type()
        if _ <= trials // 2:
            run_game(vector, minmax_agent, second_agent)
        else:
            run_game(vector, second_agent, minmax_agent)
        scores.append(sum(minmax_agent.numbers))
    visualize_distribution(scores, depth)


def visualize_distribution(scores, depth):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Scores for MinMaxAgent (Depth {depth})")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Test for GreedyAgent
    for depth in [2, 15]:
        test_agent(GreedyAgent, depth)

    # Test for NinjaAgent
    for depth in [2, 15]:
        test_agent(NinjaAgent, depth)

    # Test for MinMaxAgent(15)
    for depth in [2, 15]:
        test_agent(lambda: MinMaxAgent(15), depth)

    # Test for RandomAgent
    for depth in [2, 15]:
        test_agent(RandomAgent, depth)
