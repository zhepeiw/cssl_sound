import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CasslePredictor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bottleneck_size,
        output_size,
        norm_type='bn',
    ):
        super(CasslePredictor, self).__init__()
        if norm_type == 'bn':
            norm1 = nn.BatchNorm1d(hidden_size)
            norm2 = nn.BatchNorm1d(hidden_size)
            norm3 = nn.BatchNorm1d(bottleneck_size)
        elif norm_type == 'in':
            norm1 = nn.GroupNorm(hidden_size, hidden_size)
            norm2 = nn.GroupNorm(hidden_size, hidden_size)
            norm3 = nn.GroupNorm(bottleneck_size, bottleneck_size)
        elif norm_type == 'ln':
            norm1 = nn.GroupNorm(1, hidden_size)
            norm2 = nn.GroupNorm(1, hidden_size)
            norm3 = nn.GroupNorm(1, bottleneck_size)
        elif norm_type == 'id':
            norm1 = nn.Identity()
            norm2 = nn.Identity()
            norm3 = nn.Identity()
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            norm1,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            norm2,
            nn.Linear(hidden_size, bottleneck_size, bias=False),
            norm3,
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_size, output_size),
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


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class PANN_Classifier(nn.Module):
    def __init__(
        self,
        input_size,
        output_size
    ):
        super(PANN_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        #  embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc2(x)
        return clipwise_output.unsqueeze(1)
