from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    # state_dim is the dimension of the input layer
    # action_dim is the dimension of the output layer
    # hidden_dim is the dimension of the hidden layer
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # define the layers

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    # forward() takes the input state (x), and passes it through the initialised layers
    # then returns the output (the predicted Q values which represent every possible action the agent can take)
    # You do not call forward() yourself, pytorch does that
    def forward(self, x):
        # relu -> Rectified Linear Unit, a common activation function
        # F.relu is a simple function that turns negative neuron outputs to zero.
        # This ensures 'noise' from currently irrelevant neurons does not affect decisions, and ensures
        # that the final decision is only driven by neurons that have actually spotted something relevant.
        x = F.relu(self.fc1(x))

        return self.fc2(x)
