import torch.nn as nn


class HiddenLayer(nn.Module):

    def __init__(self, layer_size, act_fn):
        super(HiddenLayer, self).__init__()

        if act_fn == 'ReLU':
            self.layer = nn.Sequential(
                nn.Linear(layer_size, layer_size),
                nn.ReLU())
        elif act_fn == 'LeakyReLU':
            self.layer = nn.Sequential(
                nn.Linear(layer_size, layer_size),
                nn.LeakyReLU())
        elif act_fn == 'ELU':
            self.layer = nn.Sequential(
                nn.Linear(layer_size, layer_size),
                nn.ELU())

    def forward(self, x):
        return self.layer(x)


class FCN(nn.Module):
    """
    Fully connected network
    """

    def __init__(self, output_size, input_size, hidden_size, n_layers, act_fn='LeakyReLU'):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = n_layers
        self.hidden_size = hidden_size

        if act_fn == 'ReLU':
            self.initial_layer = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU())
        elif act_fn == 'LeakyReLU':
            self.initial_layer = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LeakyReLU())
        elif act_fn == 'ELU':
            self.initial_layer = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ELU())

        self.hidden_layers_list = []

        for i in range(self.num_layers // 2):
            self.hidden_layers_list.append(
                nn.Sequential(
                    HiddenLayer(self.hidden_size, act_fn),
                    HiddenLayer(self.hidden_size, act_fn)
                )
            )

        self.hidden_layers = nn.Sequential(*self.hidden_layers_list)

        self.net = nn.Sequential(
            self.initial_layer,
            self.hidden_layers,
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        return self.net(x)
