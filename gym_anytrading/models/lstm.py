import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self,
                 num_classes,
                 input_size,
                 hidden_size,
                 num_layers,
                 act_fn='LeakyReLU',
                 device='cuda:0',
                 pos_output=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)

        if act_fn == 'LeakyReLU':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.device = device
        self.pos_output = pos_output

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn[self.num_layers - 1].view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        if self.pos_output:
            out = self.relu(out)
        return out
