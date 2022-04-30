import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

class LogSoftmaxWithProbWrapper(nn.Module):
    """
    Arguments
    ---------
    Returns
    ---------
    loss : torch.Tensor
        Learning loss
    predictions : torch.Tensor
        Log probabilities
    Example
    -------
    """

    def __init__(self, loss_fn):
        super(LogSoftmaxWithProbWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1, outdim].
        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class SimCLRLoss(nn.Module):
    '''
        Loss for SimCLR
    '''
    def __init__(self, tau=0.5):
        super(SimCLRLoss, self).__init__()
        self.tau = tau
        self.criterion = NTXentLoss(temperature=tau)

    def forward(self, z1, z2):
        """
        Arguments
        ---------
        z1 : torch.Tensor (B, D)
            Projected features of augmented examples
        z2 : torch.Tensor (B, D)
            Projected features of the same examples with different augmentations
        Returns
        ---------
        loss : torch.Tensor 
            Scalar NT-Xent loss
        """
        z_pairs = torch.cat([z1, z2], dim=0) # (2B, D)
        indices = torch.arange(0, z1.shape[0], device=z1.device)
        labels = torch.cat([indices, indices], dim=0)

        return self.criterion(z_pairs, labels)

class BarlowTwinsLoss(nn.Module):
    '''
        Loss for Barlow Twins
        Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    '''
    def __init__(self, lambda_rr=5e-3, out_dim=8192, eps=1e-8, loss_scale=1.):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_rr = lambda_rr
        self.out_dim = out_dim
        self.eps = eps
        self.loss_scale = loss_scale

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        """
        Arguments
        ---------
        z1 : torch.Tensor (B, D)
            Projected features of augmented examples
        z2 : torch.Tensor (B, D)
            Projected features of the same examples with different augmentations
        Returns
        ---------
        loss : torch.Tensor 
            Scalar Barlow Twins loss
        """ 
        B, D = z1.shape
        z_1_norm = (z1 - z1.mean(0)) / (z1.std(0) + self.eps)
        z_2_norm = (z2 - z2.mean(0)) / (z2.std(0) + self.eps)
        c = z_1_norm.T @ z_2_norm / B

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_rr * off_diag
        return self.loss_scale * loss
        # return loss / (c.shape[0]*c.shape[1]) * self.loss_scale
