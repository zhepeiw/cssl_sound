from speechbrain.utils import checkpoints
import math
import torch
import numpy as np
import pdb


@checkpoints.register_checkpoint_hooks
class SimSiamCosineScheduler:
    """
        todo
    """

    def __init__(
        self,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        steps_per_epoch,
        constant_predictor_lr=False,
    ):
        final_lr = float(final_lr)
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        self.warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_steps)
        cosine_steps = int(steps_per_epoch * (num_epochs - warmup_epochs))
        self.cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(cosine_steps)/cosine_steps))

        #  self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))


        self.n_steps = 0

    def on_batch_end(self, opt):
        """
        Arguments
        ---------
        opt : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.
        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        for param_group in opt.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = self._get_lr_scale(self.n_steps)
        self.n_steps += 1

    def __call__(self, epoch):
        old_lr = self._get_lr_scale(self.n_steps-1)
        new_lr = self._get_lr_scale(self.n_steps)

        return old_lr, new_lr

    def _get_lr_scale(self, step):
        warmup_steps = len(self.warmup_lr_schedule)
        if step < warmup_steps:
            return self.warmup_lr_schedule[max(0, step)]
        else:
            idx = (step - warmup_steps) % len(self.cosine_lr_schedule)
            return self.cosine_lr_schedule[idx]

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {
            "base_lr": self.base_lr,
            "warmup_lr_schedule": self.warmup_lr_schedule,
            "cosine_lr_schedule": self.cosine_lr_schedule,
            "n_steps": self.n_steps
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device  # Unused here
        data = torch.load(path)
        self.n_steps = data["n_steps"]
        self.base_lr = data["base_lr"]
        self.warmup_lr_schedule = data["warmup_lr_schedule"]
        self.cosine_lr_schedule = data["cosine_lr_schedule"]


if __name__ == "__main__":
    import torch.nn as nn
    model = nn.Linear(3, 4)
    scheduler = SimSiamCosineScheduler(
        10, 0, 100, 0.01, 1e-6, 10,
    )
    optim = torch.optim.SGD(model.parameters(), lr=9)
    pdb.set_trace()
    for i in range(2000):
        scheduler.on_batch_end(optim)
        print(i, optim.param_groups[0]['lr'])
