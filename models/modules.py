import torch
import torch.nn as nn
import pdb


class SimSiamPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, norm_type='bn'):
        super(SimSiamPredictor, self).__init__()
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type == 'in':
            norm = nn.GroupNorm(hidden_size, hidden_size)
        elif norm_type == 'ln':
            norm = nn.GroupNorm(1, hidden_size)
        elif norm_type == 'id':
            norm = nn.Identity()
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            norm,
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
        norm_type='bn',
    ):
        super(SimSiamProjector, self).__init__()
        if norm_type == 'bn':
            norm1 = nn.BatchNorm1d(hidden_size)
            norm2 = nn.BatchNorm1d(hidden_size)
        elif norm_type == 'in':
            norm1 = nn.GroupNorm(hidden_size, hidden_size)
            norm2 = nn.GroupNorm(hidden_size, hidden_size)
        elif norm_type == 'ln':
            norm1 = nn.GroupNorm(1, hidden_size)
            norm2 = nn.GroupNorm(1, hidden_size)
        elif norm_type == 'id':
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            norm1,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            norm2,
        )

    def forward(self, x):
        '''
            input: [B, 1, D]
        '''
        return self.net(x.squeeze(1)).unsqueeze(1)

class Classifier(nn.Module):
    def __init__(
        self,
        input_size,
        output_size
    ):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x.squeeze(1)).unsqueeze(1)
