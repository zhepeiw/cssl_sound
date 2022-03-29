import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class TL_Logmel(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        norm_type
    ):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm0 = nn.BatchNorm2d(mel_bins)
        elif norm_type == 'in':
            #  self.norm0 = nn.GroupNorm(mel_bins, mel_bins)
            self.norm0 = nn.InstanceNorm2d(mel_bins, affine=True, track_running_stats=True)
        elif norm_type == 'ln':
            self.norm0 = nn.GroupNorm(1, mel_bins)
        elif norm_type == 'id':
            self.norm0 = nn.Identity()
        else:
            raise ValueError('Unknown norm type {}'.format(norm_type))
        self.init_weight()

    def init_weight(self):
        init_bn(self.norm0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.norm0(x)
        x = x.transpose(1, 3)
        return x
