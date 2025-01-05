import copy
import os
import pickle
import pygame
import time

import torch

from food import Food
from model import game_state_to_data_sample, MLP
from snake import Snake, Direction



def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # agent = HumanAgent(block_size, bounds)  # Once your agent is good to go, change this line
    # agent = MLPAgent("model_ReLU_2_layers.pth", block_size, bounds)
    agent = MLPAgent("model_ReLU_1_layer_64_neurons_drop_reg.pth", block_size, bounds)
    # agent = MLPAgent("model_ReLU_1_layer_64_neurons_drop_reg.pth", block_size, bounds)
    scores = []
    games = 0
    run = True
    pygame.time.delay(1000)
    while run:
        #speed
        pygame.time.delay(60)  # Adjust game speed, decrease to test your agent and model quickly

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
    agent.dump_data()
    pygame.quit()


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


class MLPAgent:
    def __init__(self, model_path, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.model = torch.load(model_path)

    def act(self, game_state) -> Direction:
        data_sample = game_state_to_data_sample(game_state, self.block_size, self.bounds)
        data_sample = data_sample.reshape(1, -1)
        data_sample = torch.tensor(data_sample).float()
        output = self.model(data_sample)

        # Prevent going back
        current_direction = game_state["snake_direction"]
        if current_direction == Direction.UP:
            output[0][Direction.DOWN.value] = -1
        elif current_direction == Direction.DOWN:
            output[0][Direction.UP.value] = -1
        elif current_direction == Direction.LEFT:
            output[0][Direction.RIGHT.value] = -1
        elif current_direction == Direction.RIGHT:
            output[0][Direction.LEFT.value] = -1

        action = torch.argmax(output).item()
        return Direction(action)
    def dump_data(self):
        # Metoda nie jest potrzebna dla tego agenta, ale jest zaimplementowana, aby zachować spójność.
        pass


if __name__ == "__main__":
    main()
