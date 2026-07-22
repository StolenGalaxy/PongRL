from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        # deque -> "double ended queue"
        # a deque is simply a list that is optimised for accessing/removing items near the start and end
        # with a maxlen, it will automatically remove items at the start when new items are added past it's maxlen
        self.memory = deque(maxlen=maxlen)

        if seed is not None:
            # random.seed() sets a seed for the random number generator to allow for reproducibility if we need it
            random.seed(seed)

    # a transition is a single experience
    # it contains a state, action, next_state, and reward
    # we will define the transition as a tuple elsewhere
    def append(self, transition):
        self.memory.append(transition)

    # retrieve a number of random elements from memory, which we will then use for training
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
