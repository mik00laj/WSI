import copy
import os
import pickle
import pygame
import time
import numpy as np

from food import Food
from snake import Snake, Direction
from NaiveBayes import NaiveBayes, load_data, game_state_to_data_sample

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
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]},
                        f)  # Last 10 frames are when you press exit, so they are bad, skip them

class BehavioralCloningAgent:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return np.array([self.model.predict(x.reshape(1, -1))[0] for x in X])

    def act(self, game_state) -> Direction:
        data_sample = game_state_to_data_sample(game_state)
        prediction = self.model.predict([data_sample])[0]
        return Direction(prediction)

def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # Load trained model
    X, y = load_data()
    model = NaiveBayes()
    agent = BehavioralCloningAgent(model)
    agent.fit(X, y)

    scores = []
    games = 0
    run = True
    pygame.time.delay(1000)
    while run:
        pygame.time.delay(0)

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
