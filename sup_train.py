import os
import sys
import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import logging
import datetime
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from dataset.prepare_urbansound8k import prepare_split_urbansound8k_csv
from dataset.data_pipelines import dataio_prep

import pdb


class SupSoundClassifier(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # TODO: augmentation

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)

        if self.hparams.amp_to_db:
            Amp2db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=80
            )  # try "magnitude" Vs "power"? db= 80, 50...
            feats = Amp2db(feats)

        # Normalization
        if self.hparams.normalize:
            feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + sound classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        predictions, lens = predictions  # [bs, 1, C]
        targets, _ = batch['class']  # [bs, 1]
        # TODO: augmentation
        loss = self.hparams.compute_cost(predictions, targets, lens)
        # TODO: lr scheduler
        if hasattr(self.hparams.lr_scheduler, "on_batch_end"):
            self.hparams.lr_scheduler.on_batch_end(self.optimizer)
        # TODO: metrics
        self.acc_metric.append(predictions, targets, lens)
        return loss

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.
        The default implementation depends on a few methods being defined
        with a particular behavior:
        * ``compute_forward()``
        * ``compute_objectives()``
        Also depends on having optimizers passed at initialization.
        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig.data.shape[0])
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.optimizer.step()
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig.data.shape[0])
            self.optimizer.zero_grad()

        # wandb logger
        if self.hparams.use_wandb:
            self.train_loss_buffer.append(loss.item())
            if self.step % self.hparams.train_log_frequency == 0 and self.step > 1:
                self.hparams.train_logger.log_stats(
                    stats_meta={"datapoints_seen": self.hparams.datapoint_counter.current},
                    train_stats={'buffer-loss': np.mean(self.train_loss_buffer)},
                )
                self.train_loss_buffer = []

        return loss.detach().cpu()

    def check_gradients(self, loss):
        """Check if gradients are finite and not too large.

        Automatically clips large gradients.

        Arguments
        ---------
        loss : tensor
            The loss tensor after ``backward()`` has been called but
            before the optimizers ``step()``.

        Returns
        -------
        bool
            Whether or not the optimizer step should be carried out.
        """
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Print helpful debug info
            logger.warn(f"Loss is {loss}.")
            for p in self.modules.parameters():
                if not torch.isfinite(p).all():
                    logger.warn("Parameter is not finite: " + str(p))

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warn("Patience not yet exhausted, ignoring this batch.")
                return False

        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.modules.parameters()), self.max_grad_norm
        )

        return True

    def on_fit_start(self):
        super().on_fit_start()
        # wandb logger
        self.train_loss_buffer = []
        self.train_stats = {}

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.acc_metric = self.hparams.acc_metric()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage == sb.Stage.TRAIN:
            self.train_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }
        elif stage == sb.Stage.VALID:
            valid_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }
        else:
            test_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "datapoints_seen": self.hparams.datapoint_counter.current,
                },
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta={'acc': valid_stats['acc']}, max_keys=["acc"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current,},
                test_stats=test_stats
            )


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # setting up experiment stamp
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d+%H-%M-%S')
    if run_opts['debug']:
        time_stamp = 'debug_' + time_stamp
    stamp_override = 'time_stamp: {}'.format(time_stamp)
    overrides = stamp_override + '\n' + overrides if len(overrides) > 0 else stamp_override

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger_fn']()

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # csv files split by task
    run_on_main(
        prepare_split_urbansound8k_csv,
        kwargs={
            "root_dir": hparams["data_folder"],
            'output_dir': hparams['save_folder'],
            'task_classes': hparams['task_classes'],
            'train_folds': hparams['train_folds'],
            'valid_folds': hparams['valid_folds'],
            'test_folds': hparams['test_folds'],
        }
    )

    train_data = dataio_prep(
        hparams,
        os.path.join(hparams['save_folder'], 'train_task0_raw.csv'),
        #  hparams['train_duration']
    )
    valid_data = dataio_prep(
        hparams,
        os.path.join(hparams['save_folder'], 'valid_task0_raw.csv'),
    )

    brain = SupSoundClassifier(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["train_dataloader_opts"],
    )
