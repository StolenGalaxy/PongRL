import torch

from game import Game
from dqn import DQN

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def run(self, training=True, render=True):
        game = Game(render)

        # the policy dqn is the active, working network that actually decides the agent's current actions
        # (decides it's "policy")
        policy_dqn = DQN(4, 3).to(device)

        action = 0
        while True:
            reward = game.play_step(action)
