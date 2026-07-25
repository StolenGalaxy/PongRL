from datetime import datetime
import os
import random

import torch
from torch import nn

from game import Game
from dqn import DQN

from experience_replay import ReplayMemory

import itertools

import yaml

# Folder to store runs and trained model in
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[self.hyperparameter_set]

        self.memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.model_save_rate = hyperparameters["model_save_rate_games"]

        # how much should the network adjust parameters at each step of optimisation
        # if it 'learns' too quickly, it may learn incorrectly
        self.learning_rate_alpha = hyperparameters["learning_rate_alpha"]

        # how much should the network value immediate rewards over future ones
        self.discount_factor_gamma = hyperparameters["discount_factor_gamma"]

        # use MSE as the loss function
        self.loss_fn = nn.MSELoss()

        # we set the optimiser later
        self.optimiser = None

        # store data
        self.MODEL_FILE = os.path.join(MODEL_DIR, f"{self.hyperparameter_set}.pt")


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

            target_dqn = DQN(5, 3).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0

            # use the Adam optimiser (gradient descent is another example of an optimiser)
            # we pass in policy_dqn.parameters(), essentially providing Adam the memory references to the weights and
            # biases for it to update later when we call self.optimiser.step()
            self.optimiser = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_alpha)

            # track our highest reward
            highest_reward = -99999999
        else:
            # load our saved model and set our policy network to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        rewards_per_episode = []
        epsilon_history = [self.epsilon_init]

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

                    memory.append((state, action_tensor, new_state, reward_tensor, terminated))
                    step_count += 1

                    if len(memory) > self.mini_batch_size:
                        # retrieve a batch of samples from memory
                        mini_batch = memory.sample(self.mini_batch_size)

                        self.optimise(mini_batch, policy_dqn, target_dqn)

                        # if enough steps have been taken, sync the networks
                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

            rewards_per_episode.append(episode_reward)

            if training:
                if episode_reward > highest_reward:
                    # if a new highest reward is achieved, log it, and ensure the model is saved
                    print(f"{datetime.now()}: New highest reward: {episode_reward}")
                    highest_reward = episode_reward

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                elif not episode % self.model_save_rate:
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                # decrease epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

    def optimise(self, mini_batch, policy_dqn, target_dqn):
        # we calculate the predicted q value (our current guess) and the target q value
        # then we use a loss function (in this case, MSE) to calculate the difference between the guess and the target
        # then use an optimiser (in this case, Adam) to adjust the policy network's weights so it's next guess
        # will be closer to the target
        states, actions, new_states, rewards, terminations = zip(*mini_batch)


        # why do we use torch.cat?
        # for example, for the states:
        # each state is represented by a tensor containing 5 different values
        # so states is essentially 32 * [5]
        # however, for pytorch to process these states, we want them to be one big [32, 5] tensor
        # this is what torch.cat (concatenate) does. It groups them all into one big tensor.
        states = torch.cat(states)
        actions = torch.cat(actions).long()
        new_states = torch.cat(new_states)
        rewards = torch.cat(rewards)

        terminations = torch.tensor(terminations).float().to(device)


        with torch.no_grad():
            # if a termination is true (1), target_q = reward + 0 * ... = reward
            # this is because if the game was terminated, there are no future rewards

            # this is the Bellman Equation
            # Q(s, a) = R + discount factor (gamma) * maximum of all future rewards

            # remember we are performing this for every experience at once
            target_q = rewards + (1-terminations) * self.discount_factor_gamma * target_dqn(new_states).max(dim=1)[0]


        # current_q is the policy network's guess of how valuable each action is
        # target_q is the target network's more accurate calculation of what it was worth
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # calculate loss (difference between the predicted reward for each action and the actual reward)
        # the loss function we are using is MSE (mean squared error)
        loss = self.loss_fn(current_q, target_q)

        self.optimiser.zero_grad() # Clear any gradients from the optimiser

        loss.backward() # Now we finally have the loss, the difference between our estimated q and the target q
        # We don't actually need this value for any calculation. We instead call loss.backward(). This goes back
        # through the entire calculation that was performed to reach it, and records the gradients used.

        self.optimiser.step() # Finally, with these new gradients, update the network parameters (weights and biases)
        # to ensure predicted q values are more accurate in the future

Agent("oneplayerpong").run()
