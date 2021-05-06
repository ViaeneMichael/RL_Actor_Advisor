import torch
import torch.nn as nn

class SeaquestNet(nn.Module):
    """
    Neural network with...
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
