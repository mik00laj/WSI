import random
import matplotlib.pyplot as plt
import pygame
import torch
import pickle

from food import Food
from snake import Snake, Direction


def main(epsilon=0.00, discount=0.80, lr=0.01, train=False):
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    # 1) TRENOWANIE is_training=train, load_qfunction_path=None
    # agent = QLearningAgent(block_size, bounds, epsilon=epsilon, discount=discount, lr=lr, is_training=True, load_qfunction_path=None)
    # 2) TESTOWANIE is_training=False, load_qfunction_path='_function_epsX.XXX_discountX.XX_lrXXX_train.pkl'
    agent = QLearningAgent(block_size, bounds, epsilon=epsilon, discount=discount, lr=lr, is_training=False, load_qfunction_path="q_function_eps0.001_discount0.99_lr0.01_train.pkl")
    scores = []
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode, total_episodes = 0, 100
    while episode < total_episodes and run:
        print(episode)
        # pygame.time.delay(1)  # Adjust game speed, decrease to learn agent faster

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction,
                      "bounds": bounds}

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
    print(f"Mean score value: {sum(scores)/len(scores)}   Max score value: {max(scores)}")
    # This will create a smoothed mean score per episode plot.
    # I want you to create a smoothed sum of  rewards per episode plots, that's how we evaluate RL algorithms!
    scores = torch.tensor(scores, dtype=torch.float).unsqueeze(0)
    scores = torch.nn.functional.avg_pool1d(scores, 31, stride=1)
    plt.plot(scores.squeeze(0))
    plt.ylim(0, 20)
    plt.savefig(f"mean_score_eps{epsilon}_discount{discount}_lr{lr}_{'train' if train else 'test'}.png")
    print("Check out mean_score.png")
    agent.dump_qfunction()
    pygame.quit()


class QLearningAgent:
    def __init__(self,
                 block_size,
                 bounds,
                 epsilon=0.1,
                 discount=0.9,
                 is_training=True,
                 load_qfunction_path=None,
                 lr=0.01,
                 qtable_size=(2, 2, 2, 2, *(2 for i in range(16)), 4, 4)
                 ):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.is_training = is_training
        if load_qfunction_path is not None:
            with open(load_qfunction_path, 'rb') as file:
                self.Q = pickle.load(file)
        else:
            self.Q = torch.zeros(qtable_size)
        self.obs = None
        self.action = None
        self.eps = epsilon
        self.discount = discount
        self.learning_rate = lr

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        new_obs = self.game_state_to_observation(game_state)
        new_action = random.randint(0, 3)

        if random.random() > self.eps:
            new_action = torch.argmax(self.Q[new_obs])

        if self.action is not None:
            # Calculate the Q-value of the new state for all actions
            max_q_value_new_state = torch.max(self.Q[new_obs])

            # Update Q-value for the current state-action pair
            if not is_terminal:
                update = reward + self.discount * max_q_value_new_state - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]

            self.Q[self.obs][self.action] += self.learning_rate * update

        self.action = new_action
        self.obs = new_obs
        return Direction(int(new_action))

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        new_obs = self.game_state_to_observation(game_state)
        new_action = random.randint(0, 3)

        if random.random() > self.eps:
            new_action = torch.argmax(self.Q[new_obs])

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

        # Define relative coordinates of each adjacent to head field
        relative_coords = [(-30, -30), (0, -30), (30, -30), (30, 0),
                           (30, 30), (0, 30), (-30, 30), (-30, 0)]
        head_x, head_y = gs["snake_body"][-1]

        tail = [0] * 8
        # Check for each adjacent field if there is a tail
        for i, (dx, dy) in enumerate(relative_coords):
            if (head_x + dx, head_y + dy) in gs["snake_body"][:-1]:
                tail[i] = 1

        wall = [0] * 8
        bounds_x, bounds_y = gs["bounds"]
        for i, (dx, dy) in enumerate(relative_coords):
            if (head_x + dx < 0) or (head_x + dx >= bounds_x) or \
                    (head_y + dy < 0) or (head_y + dy >= bounds_y):
                wall[i] = 1

        return (is_up, is_right, is_down, is_left, *tail, *wall, gs["snake_direction"].value)

    def dump_qfunction(self):
        filename = f"q_function_eps{self.eps}_discount{self.discount}_lr{self.learning_rate}_{'train' if self.is_training else 'test'}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)


if __name__ == "__main__":
    main()