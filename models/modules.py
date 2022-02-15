import torch
import torch.nn as nn
import pdb


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
        '''
            input: [B, 1, D]
        '''
        return self.net(x.squeeze(1)).unsqueeze(1)


class SimSiamProjector(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
    ):
        super(SimSiamProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size)
        )

    def forward(self, x):
        '''
            input: [B, 1, D]
        '''
        return self.net(x.squeeze(1)).unsqueeze(1)

