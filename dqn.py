from torch import nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    # state_dim is the dimension of the input layer
    # action_dim is the dimension of the output layer
    # hidden_dim is the dimension of the hidden layer(s)
    def __init__(self, state_dim, action_dim, enable_dueling=True, hidden_dim=32):
        super().__init__()

        self.enable_dueling = enable_dueling

        # define the layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # Dueling splits the Q value calculation for each action into two streams
        if self.enable_dueling:
            # why use dueling dqn?

            # in reinforcement learning, for the majority of states, the specific action an agent takes will not make
            # much difference to the overall result (for example, if a game is 60 frames per second, a single
            # frame's action won't change much).

            # splitting q value calculations allows for a separate stream for determining whether a state is good,
            # and a separate stream for determining whether an action is good - with each stream having its own weights
            # and biases

            # how does this help?

            # essentially, because we have a specific stream for determining whether a state itself is good or bad,
            # we can more quickly learn to avoid actions that lead us to bad states


            # Value stream - predicts how good it is to be in this exact state regardless of the action you take next
            self.fc_value_1 = nn.Linear(hidden_dim, hidden_dim) # each stream can have hidden layers
            self.fc_value_2 = nn.Linear(hidden_dim, 1) # the final output of the value stream is a single
            # value of how good that exact state is


            # Advantages stream - predicts how much better or worse each action is compared to the average action
            # in that state
            self.fc_advantage_1 = nn.Linear(hidden_dim, hidden_dim) # again, each stream can have hidden layers
            self.fc_advantage_2 = nn.Linear(hidden_dim, action_dim) # the final output of the advantage stream is
            # a value for how good each action is
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    # forward() takes the input state (x), and passes it through the initialised layers
    # then returns the output (the predicted Q values which represent every possible action the agent can take)
    # You do not call forward() yourself, pytorch does that
    def forward(self, x):
        # relu -> Rectified Linear Unit, a common activation function
        # F.relu is a simple function that turns negative neuron outputs to zero.
        # This ensures 'noise' from currently irrelevant neurons does not affect decisions, and ensures
        # that the final decision is only driven by neurons that have actually spotted something relevant.
        x = F.relu(self.fc1(x))

        if self.enable_dueling:
            # Value calculation
            v = F.relu(self.fc_value_1(x))
            v = self.fc_value_2(v)

            # Advantage calculation
            a = F.relu(self.fc_advantage_1(x))
            a = self.fc_advantage_2(a)

            # the mean advantage of all actions for a state must be zero, so we subtract the mean itself from each
            # advantage, to ensure the mean is zero (eg think 1, 2, and 3 -> subtract mean (2) from each -> -1, 0, 1 ->
            # new mean is zero)
            a = a - torch.mean(a, dim=1, keepdim=True)

            # finally, we add the value for that state to each action's value to get a Q value for each action in that
            # state
            Q = v + a
        else:
            x = F.relu(self.fc2(x))
            Q = self.fc3(x)
        return Q
