import torch
import torch.nn as nn
from speechbrain.lobes import augment
import pdb


class TFAugmentation(nn.Module):
    def __init__(
        self,
        time_warp=True,
        time_warp_window=5,
        time_warp_mode="bicubic",
        freq_mask=True,
        freq_mask_width=(0, 20),
        n_freq_mask=2,
        time_mask=True,
        time_mask_width=(0, 100),
        n_time_mask=2,
        replace_with_zero=True,
    ):
        super().__init__()
        self.specaugment = augment.SpecAugment(
            time_warp=time_warp,
            time_warp_window=time_warp_window,
            time_warp_mode=time_warp_mode,
            freq_mask=freq_mask,
            freq_mask_width=freq_mask_width,
            n_freq_mask=n_freq_mask,
            time_mask=time_mask,
            time_mask_width=time_mask_width,
            n_time_mask=n_time_mask,
            replace_with_zero=replace_with_zero
        )
    def forward(self, x, lens):
        x = self.specaugment(x)
        return x


