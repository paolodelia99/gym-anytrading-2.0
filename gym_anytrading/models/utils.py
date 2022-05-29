import torch
import torch.nn as nn


@torch.no_grad()
def init_xuniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)