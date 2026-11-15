import torch
import torch.nn as nn


class DeepLinear(nn.Module):

    def __init__(self, d:int=256, depth:int=10,):
        super().__init__()

        self.layers = nn.ParameterList([nn.Parameter(torch.empty(d, d)) for _ in range(depth)])

        self.reset_parameters()

    def reset_parameters(self):

        for W in self.layers:
            nn.init.orthogonal_(W)

    def forward(self):
        W = self.layers[0]
        for layer in self.layers[1:]:
            W = layer @ W
        return W