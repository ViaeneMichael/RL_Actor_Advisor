import os.path

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SeaquestNet(nn.Module):
    """
    Neural network with...
    """
    def __init__(self, learning_rate, input_shape,  chkpt_dir='tmp/ppo/actor_seaquest_ppo'):
        # Images will be 84*84*3, a stack 3
        super(SeaquestNet, self).__init__()
        self.input_shape = input_shape
        self.checkpoint_file = os.path.normpath(chkpt_dir)
        print(self.checkpoint_file)
        self.lr = learning_rate
        self.body = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()  # not sure about this
        )

        self.value = nn.Sequential( # value of state
            nn.Linear(5776, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )

        self._policy = nn.Sequential( # probabilities
            nn.Linear(5776, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 18),
            nn.Softmax(-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)


    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    def convOutput(self, state):
        return self.body(state)

    def policy(self, state):
        return self._policy(self.convOutput(state))

    def stateValue(self, state):
        return self.value(self.convOutput(state))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

