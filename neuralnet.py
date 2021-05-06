import torch
import torch.nn as nn
import torch.nn.functional as F

class SeaquestNet(nn.Module):
    """
    Neural network with...
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Images will be 84*84*3, a stack 3
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*20*20, 120) # ((84-3+1)/2-3+1)/2=20 (W-F+2P)/S+1
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,18)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*20*20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.Softmax(-1)(x)
        return x