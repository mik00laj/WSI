import random
import matplotlib.pyplot as plt
import pygame
import torch

from food import Food
from snake import Snake, Direction


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    agent = QLearningAgent(block_size, bounds, 0.2, 0.99, is_training=True)
    scores = []
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode, total_episodes = 0, 1000
    while episode < total_episodes and run:
        # pygame.time.delay(1)  # Adjust game speed, decrease to learn agent faster

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state, reward, is_terminal)
        reward = -0.001
        is_terminal = False
        snake.turn(direction)
        snake.move()
        reward += snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(1)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= 0.999
            is_terminal = True

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    # This will create a smoothed mean score per episode plot.
    # I want you to create a smoothed sum of  rewards per episode plots, that's how we evaluate RL algorithms!
    scores = torch.tensor(scores, dtype=torch.float).unsqueeze(0)
    scores = torch.nn.functional.avg_pool1d(scores, 31, stride=1)
    plt.plot(scores.squeeze(0))
    plt.savefig("mean_score.png")
    print("Check out mean_score.png")
    agent.dump_qfunction()
    pygame.quit()


class QLearningAgent:
    def __init__(self, block_size, bounds, epsilon, discount, is_training=True, load_qfunction_path=None):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.is_training = is_training
        self.Q = torch.zeros((2, 2, 2, 2, 4, 4))
        self.obs = None
        self.action = None

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        # TODO: There are many hardcoded hyperparameters here, what do they do? Replace them with names.
        new_obs = self.game_state_to_observation(game_state)

        new_action = random.randint(0, 3)
        if random.random() > 0.1:
            new_action = torch.argmax(self.Q[new_obs])

        if self.action is not None:
            if not is_terminal:
                update = reward + 0.9 * torch.max(self.Q[new_obs]) - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]
            self.Q[self.obs][self.action] += 0.01 * update

        self.action = new_action
        self.obs = new_obs
        return Direction(int(new_action))

    @staticmethod
    def game_state_to_observation(game_state):
        gs = game_state
        is_up = int(gs["food"][1] < gs["snake_body"][-1][1])
        is_right = int(gs["food"][0] > gs["snake_body"][-1][0])
        is_down = int(gs["food"][1] > gs["snake_body"][-1][1])
        is_left = int(gs["food"][0] < gs["snake_body"][-1][0])
        return is_up, is_right, is_down, is_left, gs["snake_direction"].value

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        raise NotImplementedError()

    def dump_qfunction(self):
        raise NotImplementedError()


if __name__ == "__main__":
    main()
