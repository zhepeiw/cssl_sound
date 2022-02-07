import torch
import torch.nn as nn


class SimSiamPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimSiamPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.net(x.squeeze(1)).unsqueeze(1)
