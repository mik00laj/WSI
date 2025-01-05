import random
import time
import numpy as np
import matplotlib.pyplot as plt

random.seed(28)  # ZROBIONE


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__ (OOOO000O000O00000 ):
        OOOO000O000O00000 .numbers =[]
    def act (O000000O000OO0O0O ,O0OO0O0O0O0OO0O00 ):
        if len (O0OO0O0O0O0OO0O00 )%2 ==0 :
            O00O0O0000000OO0O =sum (O0OO0O0O0O0OO0O00 [::2 ])
            O0O00O0OO00O0O0O0 =sum (O0OO0O0O0O0OO0O00 )-O00O0O0000000OO0O
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
                return O0OO0O0O0O0OO0O00 [1 :] # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
            return O0OO0O0O0O0OO0O00 [:-1 ]
        else :
            O00O0O0000000OO0O =max (sum (O0OO0O0O0O0OO0O00 [1 ::2 ]),sum (O0OO0O0O0O0OO0O00 [2 ::2 ]))
            O0O00O0OO00O0O0O0 =max (sum (O0OO0O0O0O0OO0O00 [:-1 :2 ]),sum (O0OO0O0O0O0OO0O00 [:-2 :2 ]))
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
                return O0OO0O0O0O0OO0O00 [:-1 ]
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
            return O0OO0O0O0O0OO0O00 [1 :]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        self.numbers = []
        self.max_depth = max_depth

    def act(self, vector: list, depth=0, maximizingPlayer=True):
        if depth == self.max_depth or len(vector) == 0:
            return sum(self.numbers), []

        if maximizingPlayer:
            bestValue = float('-inf')
            bestMove = None

            for i in [0, -1]:
                if i == 0:
                    new_vector = vector[1:]
                else:
                    new_vector = vector[:-1]
                self.numbers.append(vector[i])
                value, _ = self.act(new_vector, depth+1, False)
                self.numbers.pop()

                if value > bestValue:
                    bestValue = value
                    bestMove = i

            if depth == 0:
                self.numbers.append(vector[bestMove])
                return vector[1:] if bestMove == 0 else vector[:-1]
            else:
                return bestValue, bestMove
        else:
            worstValue = float('inf')
            worstMove = None

            for i in [0, -1]:
                if i == 0:
                    new_vector = vector[1:]
                else:
                    new_vector = vector[:-1]
                value, _ = self.act(new_vector, depth+1, True)

                if value < worstValue:
                    worstValue = value
                    worstMove = i

            return worstValue, worstMove

def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)

def test_agent(depths, n_games=1000, vector_length=15): # Funkcja testowa
    results = {}
    for depth in depths:
        times = []
        scores = []
        for _ in range(n_games):
            vector = [random.randint(-10, 10) for _ in range(vector_length)]
            minmax_agent = MinMaxAgent(max_depth=depth)
            # second_agent = RandomAgent()  # MinMaxAgent vs RandomAgent
            second_agent = GreedyAgent()  # MinMaxAgent vs GreedyAgent
            # second_agent = NinjaAgent()    # MinMaxAgent vs NinjaAgent
            # second_agent = MinMaxAgent(15) # MinMaxAgent vs MinMaxAgent(15)
            start_time = time.time()
            if _ <= 500:
                run_game(vector, minmax_agent, second_agent)
            else:
                run_game(vector, second_agent,minmax_agent)
            times.append(time.time() - start_time)
            scores.append(sum(minmax_agent.numbers))

        avg_time = np.mean(times)
        avg_score = np.mean(scores)
        std_dev_score = np.std(scores)
        results[depth] = (avg_time, avg_score, std_dev_score)

    return results



def main():
    vector = [random.randint(-10, 10) for _ in range(14)]
    print(f"Vector: {vector}")
    first_agent, second_agent = MinMaxAgent(), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} "
          f"Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")



if __name__ == "__main__": # Uruchomienie funkcji testowej
      # main()
    depths = [1, 2, 3, 15]
    results = test_agent(depths)
    for depth, result in results.items():
        print(f"Głębokość: {depth}, Średni czas: {result[0]:.8f}, Średnia suma punktów: {result[1]:.2f}, Odchylenie standardowe: {result[2]:.2f}")

