import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import logging
import datetime
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from dataset.prepare_urbansound8k import prepare_split_urbansound8k_csv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import wandb
from confusion_matrix_fig import create_cm_fig
from dataset.cl_pipeline import (
    prepare_task_csv_from_replay,
    mixup_dataio_ssl_prep,
)
from schedulers import SimSiamCosineScheduler

import pdb


class SimCLR(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs1, lens = batch.sig1
        wavs2, _ = batch.sig2

        x1 = self.prepare_features(wavs1, lens, stage)
        x2 = self.prepare_features(wavs2, lens, stage)
        # Embeddings
        e1 = self.modules.embedding_model(x1)
        e2 = self.modules.embedding_model(x2)
        h1 = self.modules.projector(e1)  # [B, 1, D]
        h2 = self.modules.projector(e2)  # [B, 1, D]
        z1 = self.modules.predictor(h1)  # [B, 1, D]
        z2 = self.modules.predictor(h2)  # [B, 1, D]
        o1 = self.modules.classifier(e1) # [B, 1, C]
        #  o2 = self.modules.classifier(e2) # [B, 1, C]

        return z1, z2, h1, h2, o1, lens

    def prepare_features(self, wavs, lens, stage):
        # time domain augmentation
        #  wavs_aug = self.hparams.time_domain_aug(wavs, lens)
        #  if wavs_aug.shape[1] > wavs.shape[1]:
        #      wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
        #  else:
        #      zero_sig = torch.zeros_like(wavs)
        #      zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
        #      wavs_aug = zero_sig
        #  wavs = wavs_aug

        feats = self.modules.compute_features(wavs)  # [B, T, D]
        feats = self.hparams.spec_domain_aug(feats, lens)
        if self.hparams.amp_to_db:
            Amp2db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=80
            )  # try "magnitude" Vs "power"? db= 80, 50...
            feats = Amp2db(feats)
        # Normalization
        if self.hparams.normalize:
            feats = self.modules.mean_var_norm(feats, lens)

        return feats

    def compute_objectives(self, predictions, batch, stage):
        z1, z2, h1, h2, o1, lens = predictions
        # SSL loss
        ssl_loss = self.hparams.compute_simclr_cost(z1.squeeze(), z2.squeeze())
        # supervised loss
        if self.hparams.use_mixup and stage == sb.Stage.TRAIN:
            targets, _ = batch.label_prob
        else:
            targets, _ = batch.class_string_encoded # [bs, 1]
            targets = F.one_hot(targets, o1.shape[-1]).float()  # [bs, 1, C]
        sup_loss = self.hparams.compute_sup_cost(o1, targets, lens)
        loss = self.hparams.sup_weight * sup_loss + self.hparams.ssl_weight * ssl_loss

        loss_dict = {
            'ssl': ssl_loss,
            'sup': sup_loss,
        }

        if stage == sb.Stage.TRAIN and \
           hasattr(self.hparams.lr_scheduler, "on_batch_end"):
            self.hparams.lr_scheduler.on_batch_end(self.optimizer)
        return loss, loss_dict

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
                loss, loss_dict = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig1.data.shape[0])
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss, loss_dict = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.optimizer.step()
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig1.data.shape[0])
            self.optimizer.zero_grad()

        # wandb logger
        if self.hparams.use_wandb:
            #  self.train_loss_buffer.append(loss.item())
            if len(loss_dict) > 1:
                loss_dict['loss'] = loss
            for loss_nm, loss_val in loss_dict.items():
                if loss_nm not in self.train_loss_buffer:
                    self.train_loss_buffer[loss_nm] = []
                self.train_loss_buffer[loss_nm].append(loss_val.item())
            if self.step % self.hparams.train_log_frequency == 0 and self.step > 1:
                self.hparams.train_logger.log_stats(
                    stats_meta={"datapoints_seen": self.hparams.datapoint_counter.current},
                    #  train_stats={'buffer-loss': np.mean(self.train_loss_buffer)},
                    train_stats = {'buffer-{}'.format(loss_nm): np.mean(loss_list) for loss_nm, loss_list in self.train_loss_buffer.items()},
                )
                #  self.train_loss_buffer = []
                self.train_loss_buffer = {}

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
        #  self.train_loss_buffer = []
        self.train_loss_buffer = {}
        self.train_stats = {}

    def init_optimizers(self):
        if self.opt_class is not None:
            predictor_prefix = ('module.predictor', 'predictor')
            optim_params = [{
                'name': 'base',
                'params': [param for name, param in self.modules.named_parameters() if not name.startswith(predictor_prefix)],
                'fix_lr': False,
            }, {
                'name': 'predictor',
                'params': [param for name, param in self.modules.named_parameters() if name.startswith(predictor_prefix)],
                'fix_lr': True,
            }]
            self.optimizer = self.opt_class(optim_params)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

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
        pass

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
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.TRAIN:
            old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            if not hasattr(self.hparams.lr_scheduler, "on_batch_end"):
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "datapoints_seen": self.hparams.datapoint_counter.current,
                },
                train_stats=self.train_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                #  meta={'loss': self.train_stats['loss']}, min_keys=['loss']
            )


def dataio_ssl_prep(hparams, csv_path, label_encoder):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    def random_segment(sig, target_len):
        rstart = torch.randint(0, len(sig) - target_len + 1, (1,)).item()
        return sig[rstart:rstart+target_len]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig1", "sig2")
    def audio_pipeline(wav_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        sig, read_sr = torchaudio.load(wav_path)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        # scaling
        max_amp = torch.abs(sig).max().item()
        #  assert max_amp > 0
        scaling = 1 / max_amp * 0.9
        sig = scaling * sig

        target_len = int(hparams["train_duration"] * config_sample_rate)
        if len(sig) > target_len:
            sig1 = random_segment(sig, target_len)
            sig2 = random_segment(sig, target_len)
        else:
            sig1 = sig
            sig2 = sig.clone()
        yield sig1
        yield sig2

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_name")
    @sb.utils.data_pipeline.provides("class_name", "class_string_encoded")
    def label_pipeline(class_name):
        yield class_name
        class_string_encoded = label_encoder.encode_label_torch(class_name)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig1", "sig2", "class_string_encoded"]
    )

    return ds


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

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(hparams['label_encoder_path'])
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    num_tasks = len(hparams['task_classes'])
    brain = None
    # all buffers
    replay = {'train': []}
    # CL starts here
    start_task = 0 if not hparams['resume_interrupt'] else hparams['resume_task_idx']
    for task_idx in range(start_task, num_tasks):
        print("==> Starting task {}/{}".format(task_idx+1, num_tasks))
        if not hparams['resume_interrupt']:
            # reset epoch counter and lr scheduler
            # weights should be restored already in on_evaluate_start()
            hparams['recoverables']['lr_scheduler'] = \
                    hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
            hparams['recoverables']['epoch_counter'] = \
                    hparams['epoch_counter'] = hparams['epoch_counter_fn']()
            # set new checkpointer
            hparams['checkpointer'] = sb.utils.checkpoints.Checkpointer(
                os.path.join(hparams['save_folder'], 'task{}'.format(task_idx)),
                recoverables=hparams['recoverables']
            )
            print('==> Resetting scheduler and counter at {}'.format(hparams['checkpointer'].checkpoints_dir))
        else:
            # reload everything from the interrupted checkpoint
            # set the checkpointer here, and on_fit_start() loads the content
            assert isinstance(hparams['prev_checkpointer'], sb.utils.checkpoints.Checkpointer)
            # initialize epoch counter and lr scheduler for restore
            hparams['recoverables']['lr_scheduler'] = \
                    hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
            hparams['recoverables']['epoch_counter'] = \
                    hparams['epoch_counter'] = hparams['epoch_counter_fn']()
            hparams['checkpointer'] = hparams['prev_checkpointer']
            hparams['checkpointer'].add_recoverables(hparams['recoverables'])
            # TODO: restore any external buffer for data generation here
            if task_idx > 0:
                replay_path = os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx-1),
                    'replay.pt'
                )
                if os.path.exists(replay_path):
                    replay = torch.load(replay_path)
            print("==> Resuming from interrupted checkpointer at {}".format(hparams['checkpointer'].checkpoints_dir))
            hparams['resume_interrupt'] = False

        # TODO: generate task-wise data
        curr_train_replay = prepare_task_csv_from_replay(
            os.path.join(hparams['save_folder'], 'train_task{}_raw.csv'.format(task_idx)),
            replay['train'],
            hparams['replay_num_keep'],
        )
        replay['train'] += curr_train_replay
        if hparams['use_mixup']:
            train_data = mixup_dataio_ssl_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_task{}_replay.csv'.format(task_idx)),
                label_encoder,
            )
        else:
            train_data = dataio_ssl_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_task{}_replay.csv'.format(task_idx)),
                label_encoder,
            )
        # lr scheduler setups rely on task-wise dataloader
        if isinstance(hparams['lr_scheduler'], SimSiamCosineScheduler):
            steps_per_epoch = \
                    int(np.ceil(len(train_data) / hparams['batch_size']))
            hparams['lr_scheduler_fn'].keywords['steps_per_epoch'] = \
                    steps_per_epoch
            hparams['recoverables']['lr_scheduler'] = \
                    hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
            hparams['checkpointer'].add_recoverables(hparams['recoverables'])
            print('==> Adjusting scheduler for {} steps at {}'.format(
                steps_per_epoch, hparams['checkpointer'].checkpoints_dir))

        brain = SimCLR(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        #  with torch.autograd.detect_anomaly():
        brain.fit(
            epoch_counter=brain.hparams.epoch_counter,
            train_set=train_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
        )

        brain.checkpointer.recoverables.pop('datapoint_counter')
        #  brain.on_evaluate_start(min_key='loss')
        brain.on_evaluate_start()

        if num_tasks > 1:
            # global buffer
            torch.save(
                replay,
                os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx),
                    'replay.pt'
                )
            )
        hparams['prev_checkpointer'] = hparams['checkpointer']
