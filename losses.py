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
