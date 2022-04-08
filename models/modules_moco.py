import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from functools import partial

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.reshape(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)


class MOCOProjector(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        norm_type='sbn',
        num_splits=4,
    ):
        super(MOCOProjector, self).__init__()

        # only supporting sbn and bn for now

        self.norm_type = norm_type

        sbn = partial(SplitBatchNorm, num_splits=num_splits)

        if norm_type == 'sbn':
            self.norm1 = sbn(hidden_size)
            self.norm2 = sbn(hidden_size)
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        elif norm_type == 'bn':
            norm1 = nn.BatchNorm1d(hidden_size)
            norm2 = nn.BatchNorm1d(hidden_size)
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
        if self.norm_type == 'sbn':
            x_ = x.squeeze(1)
            x_ = self.fc1(x_)
            x_ = (x_.unsqueeze(-1)).unsqueeze(-1)
            x_ = self.norm1(x_)
            x_ = (x_.squeeze(-1)).squeeze(-1)
            x_ = F.relu_(x_)
            x_ = self.fc2(x_)
            x_ = (x_.unsqueeze(-1)).unsqueeze(-1)
            x_ = self.norm2(x_)
            x_ = (x_.squeeze(-1)).squeeze(-1)

            return x_.unsqueeze(1)
        
        else:
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
