import torch
import torch.nn as nn
import torch.nn.functional as F


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
