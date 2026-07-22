import torch

from game import Game
from dqn import DQN

from experience_replay import ReplayMemory

import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

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

    def run(self, training=True, render=True):
        game = Game(render)

        # the policy dqn is the active, working network that actually decides the agent's current actions
        # (decides it's "policy")
        policy_dqn = DQN(4, 3).to(device)

        if training:
            memory = ReplayMemory(10000)

        rewards_per_episode = []

        # an episode is essentially one game
        # itertools.count is essentially a nice way of an infinite loop where we can track what number we are on
        for episode in itertools.count():
            terminated = False
            game.reset()

            episode_reward = 0

            while not terminated:
                # choose an action
                action = 1

                # get the current state
                state = self.get_state(game)
                print(state)

                # perform the action
                reward, terminated = game.play_step(action)

                # add to episode reward
                episode_reward += reward

                # get the new state
                new_state = self.get_state(game)

                if training:
                    memory.append((state, action, new_state, reward))

            rewards_per_episode.append(episode_reward)

Agent().run()
