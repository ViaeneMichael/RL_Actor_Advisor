import os.path

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SeaquestNet(nn.Module):
    """
    Neural network with...
    """
    def __init__(self, learning_rate, chkpt_dir='tmp/ppo'):
        # Images will be 84*84*3, a stack 3
        super(SeaquestNet, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_seaquest_ppo')
        self.lr = learning_rate
        self.body = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()  # not sure about this
        )

        self.value = nn.Sequential( # value of state
            nn.Linear(16*20*20, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )

        self.policy = nn.Sequential( # probabilities
            nn.Linear(16 * 20 * 20, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 18),
            nn.Softmax(-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.body(x)
        x = self.policy(x)
        return x

    # https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
    def convOutput(self, state):
        return self.body(state)

    def policy(self, state):
        return Categorical(self.policy(self.convOutput(state))) # make it a distribution to use log_prob later

    def stateValue(self, state):
        return self.value(self.convOutput(state))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))



    #     self.conv1 = nn.Conv2d(3,6,3)
    #     self.conv2 = nn.Conv2d(6,16,3)
    #     self.pool = nn.MaxPool2d(2,2)
    #     self.fc1 = nn.Linear(16*20*20, 120) # ((84-3+1)/2-3+1)/2=20 (W-F+2P)/S+1
    #     self.fc2 = nn.Linear(120,84)
    #     self.fc3 = nn.Linear(84,18)
    #
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1,16*20*20)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     x = nn.Softmax(-1)(x)
    #     return x

