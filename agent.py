import random

import torch

from game import Game
from dqn import DQN

from experience_replay import ReplayMemory

import itertools

import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]


    def get_state(self, game: Game):
        # first get the values from the game that the agent needs
        rect_x = game.rect_x
        ball_x = game.ball_x
        ball_y = game.ball_y
        ball_change_x = game.ball_change_x
        ball_change_y = game.ball_change_y

        # now we normalise them between 0 and 1 to balance their importance

        rect_x /= 700
        ball_x /= 785
        ball_y /= 600
        ball_change_x /= 5
        ball_change_y /= 5

        # place the values in a list
        raw_state = [rect_x, ball_x, ball_y, ball_change_x, ball_change_y]

        # convert them to a tensor
        # dtype (data type) sets the variables to be stored as 32-bit floating point numbers
        state_tensor = torch.tensor(raw_state, dtype=torch.float32, device=device)

        # torch.tensor.unsqueeze() creates a new dimension of size 1 at a specified position
        # we have done unsqueeze(0) meaning the new dimension is at the front
        # we have essentially changed the dimensions from [5] to [1, 5]
        # eg [x, y, z, a, b] -> [[x, y, z, a, b]]
        # this is because the neural network expects the state as a single batch of items, not a list of them
        return state_tensor.unsqueeze(0)

        # note alternatively we could replaced .unsqueeze(0) with:
        # state_tensor = torch.tensor([raw_state], dtype=...) which is generally preferred, however I have left
        # it this way to be more explicit to understand it

    def run(self, training=True, render=True):
        game = Game(render)

        # the policy dqn is the active, working network that actually decides the agent's current actions
        # (decides it's "policy")
        policy_dqn = DQN(5, 3).to(device)

        if training:
            memory = ReplayMemory(self.memory_size)

            epsilon = self.epsilon_init

        rewards_per_episode = []
        epsilon_history = []

        # an episode is essentially one game
        # itertools.count is essentially a nice way of an infinite loop where we can track what number we are on
        for episode in itertools.count():
            terminated = False
            game.reset()

            episode_reward = 0

            while not terminated:
                # get the current state
                state = self.get_state(game)

                # choose an action
                if training and random.random() < epsilon:
                    action = random.randint(0,2)
                else:
                    # by default, when passing data through a pytorch neural network, it remembers every mathematical
                    # operation so it can calculate gradients later. We are disabling that here for efficiency.
                    with torch.no_grad():
                        # get the Q values for each action, then get the index of the highest one
                        # .item() converts it from a tensor to an integer
                        action = policy_dqn(state).argmax().item()


                # perform the action
                reward, terminated = game.play_step(action)

                # add to episode reward
                episode_reward += reward

                # get the new state
                new_state = self.get_state(game)

                if training:
                    # remember, putting action and reward in [] is equivalent to performing .unsqueeze(0) later
                    action_tensor = torch.tensor([action], dtype=torch.float32, device=device)
                    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

                    memory.append((state, action_tensor, new_state, reward_tensor))

            rewards_per_episode.append(episode_reward)

            # decrease epsilon
            epsilon_history.append(epsilon)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)


Agent("oneplayerpong").run()
