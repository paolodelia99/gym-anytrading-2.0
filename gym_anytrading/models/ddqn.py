import copy

from torch import nn

from lstm import LSTM
from .utils import init_xuniform


class DDQN(nn.Module):
    """

    """

    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super().__init__()

        self.online = LSTM(output_size, input_size, hidden_size, n_layers)
        self.online.apply(init_xuniform)

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)