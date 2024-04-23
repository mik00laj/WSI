import copy
import os
import pickle
import pygame
import time

from food import Food
from model import game_state_to_data_sample, SVM, load_data
from snake import Snake, Direction
from sklearn.svm import SVC

class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)  # Last 10 frames are when you press exit, so they are bad, skip them


class BehavioralCloningAgent:
    def __init__(self, model):
        self.model = model

    def act(self, game_state) -> Direction:
        # Przekształcenie stanu gry na wektor cech
        data_sample = game_state_to_data_sample(game_state)
        predicted_action = self.model.predict([data_sample])[0]

        # Zamiana przewidzianej akcji (liczby) na kierunek
        if predicted_action == -1:
            return Direction.LEFT
        elif predicted_action == 1:
            return Direction.RIGHT
        elif predicted_action == 2:
            return Direction.UP
        elif predicted_action == 3:
            return Direction.DOWN

        # Domyślnie zwracamy kierunek w dół, jeśli coś pójdzie nie tak
        return Direction.DOWN

    def dump_data(self):
        # Metoda nie jest potrzebna dla tego agenta, ale jest zaimplementowana, aby zachować spójność.
        pass


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # Wczytywanie danych i trenowanie modelu SVM
    X, y = load_data()
    # model = SVM()
    # model = SVM(C=1)
    # model = SVC()
    model = SVC(C=1)
    # model = HumanAgent(block_size, bounds)
    model.fit(X, y)

    # Użycie agenta opartego na modelu
    agent = BehavioralCloningAgent(model)

    scores = []
    games = 0
    run = True
    pygame.time.delay(1000)
    while run:
        pygame.time.delay(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()
            games += 1
            if games >= 100:
                break


        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    # Obliczenie średniej z wyników
    avg_score = sum(scores) / len(scores)
    print(f"Avg. Score: {avg_score}")
    pygame.quit()


if __name__ == "__main__":
    main()
