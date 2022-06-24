import copy

from torch import nn

from .lstm import LSTM
from .fcn import FCN
from .utils import init_xuniform


class DDQN(nn.Module):
    """

    """

    def __init__(self, input_size, output_size, hidden_size, n_layers, recurrent: bool):
        super().__init__()

        self.recurrent = recurrent
        if self.recurrent:
            self.online = LSTM(output_size, input_size, hidden_size, n_layers)
        else:
            self.online = FCN(output_size, input_size, hidden_size, n_layers)
        self.online.apply(init_xuniform)

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
