#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : pann.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 14.02.2022
# Last Modified Date: 30.03.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

# Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from functools import partial

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, num_splits):

        super(ConvBlock, self).__init__()

        sbn = partial(SplitBatchNorm, num_splits=num_splits)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.norm_type = norm_type

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == 'in':
            #  self.norm1 = nn.GroupNorm(out_channels, out_channels)
            #  self.norm2 = nn.GroupNorm(out_channels, out_channels)
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        elif norm_type == 'ln':
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
        elif norm_type == "sbn":
            self.norm1 = sbn(out_channels)
            self.norm2 = sbn(out_channels)
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.norm1)
        init_bn(self.norm2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.norm1(self.conv1(x)))
        x = F.relu_(self.norm2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(nn.Module):
    #  def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
    #      fmax, classes_num):
    def __init__(
        self,
        mel_bins,
        emb_dim,
        norm_type='sbn',
        num_splits=4,
    ):

        super(Cnn14, self).__init__()

        #  window = 'hann'
        #  center = True
        #  pad_mode = 'reflect'
        #  ref = 1.0
        #  amin = 1e-10
        #  top_db = None

        #  # Spectrogram extractor
        #  self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #      win_length=window_size, window=window, center=center, pad_mode=pad_mode,
        #      freeze_parameters=True)
        #
        #  # Logmel feature extractor
        #  self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #      n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
        #      freeze_parameters=True)
        #
        #  # Spec augmenter
        #  self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #      freq_drop_width=8, freq_stripes_num=2)
        self.norm_type = norm_type
        self.num_splits = num_splits

        sbn = partial(SplitBatchNorm, num_splits=num_splits)

        if norm_type == 'bn':
            self.norm0 = nn.BatchNorm2d(mel_bins)
        elif norm_type == 'in':
            #  self.norm0 = nn.GroupNorm(mel_bins, mel_bins)
            self.norm0 = nn.InstanceNorm2d(mel_bins, affine=True, track_running_stats=True)
        elif norm_type == 'ln':
            self.norm0 = nn.GroupNorm(1, mel_bins)
        elif norm_type == 'sbn':
            self.norm0 = sbn(mel_bins)
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64, norm_type=norm_type, num_splits=num_splits)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, norm_type=norm_type, num_splits=num_splits)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, norm_type=norm_type, num_splits=num_splits)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, norm_type=norm_type, num_splits=num_splits)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, norm_type=norm_type, num_splits=num_splits)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=emb_dim, norm_type=norm_type, num_splits=num_splits)

        #  self.fc1 = nn.Linear(2048, 2048, bias=True)
        #  self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.norm0)
        #  init_layer(self.fc1)
        #  init_layer(self.fc_audioset)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (B, 1, T, M)"""

        #  x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        #  x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.norm0(x)
        x = x.transpose(1, 3)

        #  if self.training:
        #      x = self.spec_augmenter(x)
        #
        #  # Mixup on spectrogram
        #  if self.training and mixup_lambda is not None:
        #      x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x.unsqueeze(1)  # [B, 1, D]
        #  x = F.dropout(x, p=0.5, training=self.training)
        #  x = F.relu_(self.fc1(x))
        #  embedding = F.dropout(x, p=0.5, training=self.training)
        #  clipwise_output = torch.sigmoid(self.fc_audioset(x))
        #
        #  output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        #
        #  return output_dict


if __name__ == '__main__':
    import pdb
    x = torch.randn(32, 1, 200, 80)
    model = Cnn14(80, 2048, 'ln')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x)
    print(out.shape)
