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
#  from dataset.data_pipelines import dataio_prep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import wandb
from confusion_matrix_fig import create_cm_fig
from dataset.cl_pipeline import (
    #  prepare_task_csv_for_linclf,
    prepare_task_csv_from_replay,
    prepare_concat_csv,
    #  mixup_dataio_prep,
    class_balanced_dataio_prep,
)
from schedulers import SimSiamCosineScheduler
from cl_table_tools import compute_cl_statistics

import pdb


class LinearClassifier(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def compute_forward(self, batch, stage):
        self.modules.embedding_model.eval()
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        with torch.no_grad():
            feats = self.prepare_features(wavs, lens, stage)  # [B, T, F]
            # Embeddings + sound classifier
            embeddings = self.modules.embedding_model(feats)  # [B, 1, D]
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def prepare_features(self, wavs, lens, stage):
        # TODO: augmentation
        feats = self.modules.compute_features(wavs)
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
        predictions, lens = predictions  # [bs, 1, C]
        if self.hparams.use_mixup and stage == sb.Stage.TRAIN:
            targets, _ = batch.label_prob
        else:
            targets, _ = batch.class_string_encoded # [bs, 1]
            targets = F.one_hot(targets, predictions.shape[-1]).float()  # [bs, 1, C]
        # TODO: augmentation
        loss = self.hparams.compute_cost(predictions, targets, lens)
        if stage == sb.Stage.TRAIN and \
           hasattr(self.hparams.lr_scheduler, "on_batch_end"):
            self.hparams.lr_scheduler.on_batch_end(self.optimizer)
        # TODO: metrics
        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = targets.cpu().detach().numpy().argmax(-1).squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)

        if stage == sb.Stage.VALID:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.valid_confusion_matrix += confusion_matix
        if stage == sb.Stage.TEST:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += confusion_matix
        self.acc_metric.append(predictions, targets.argmax(-1), lens)
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
        self.acc_metric = self.hparams.acc_metric()
        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.n_classes, self.hparams.n_classes),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.n_classes, self.hparams.n_classes),
                dtype=int,
            )

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
            if not hasattr(self.hparams.lr_scheduler, "on_batch_end"):
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            # The train_logger writes a summary to stdout and to the logfile.
            # wandb logger
            if self.hparams.use_wandb:
                cm_fig = create_cm_fig(
                    self.valid_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                valid_stats.update({
                    'confusion': wandb.Image(cm_fig),
                })
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
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )
            # wandb logger
            if self.hparams.use_wandb:
                cm_fig = create_cm_fig(
                    self.test_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                test_stats.update({
                    'confusion': wandb.Image(cm_fig),
                })
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current,
                    },
                    test_stats=test_stats
                )
            else:
                self.hparams.train_logger.log_stats(
                    {
                        "Epoch loaded": self.hparams.epoch_counter.current,
                        "\n Per Class Accuracy": per_class_acc_arr_str,
                        "\n Confusion Matrix": "\n{:}\n".format(
                            self.test_confusion_matrix
                        ),
                    },
                    test_stats=test_stats,
                )

    def evaluate_multitask(
        self,
        test_datasets,
        max_key=None,
        min_key=None,
        test_loader_kwargs={},
    ):
        # do not reduce step counter for logging purposes
        self.checkpointer.recoverables.pop('datapoint_counter')
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        summary = {
            'cmat': [],
            'acc': [],
        }
        for test_data in test_datasets:
            test_loader = sb.dataio.dataloader.make_dataloader(
                test_data, **test_loader_kwargs
            )
            task_confusion_matrix = np.zeros(
                shape=(self.hparams.n_classes, self.hparams.n_classes),
                dtype=int,
            )
            # Loop over all test sentence
            with torch.no_grad():
                with tqdm(test_loader, dynamic_ncols=True) as t:
                    for i, batch in enumerate(t):
                        if self.debug and i == self.debug_batches:
                            break
                        predictions, lens = self.compute_forward(batch, stage=sb.Stage.TEST)
                        classid, _ = batch.class_string_encoded
                        y_true = classid.cpu().detach().numpy().squeeze(-1)
                        y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)
                        confusion_matix = confusion_matrix(
                            y_true,
                            y_pred,
                            labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
                        )
                        task_confusion_matrix += confusion_matix
            task_acc = np.sum(np.diag(task_confusion_matrix)) / (np.sum(task_confusion_matrix) + 1e-8)
            summary['cmat'].append(task_confusion_matrix)
            summary['acc'].append(task_acc)
        test_stats = {'avg task acc': np.mean(summary['acc'])}
        # wandb logger
        if self.hparams.use_wandb:
            import matplotlib.pyplot as plt
            # bar plot of task-wise accuracy
            acc_fig = plt.figure()
            ax = acc_fig.add_subplot(1, 1, 1)
            tickmarks = np.arange(len(summary['acc']))
            ax.bar(tickmarks, summary['acc'])
            ax.set_xlabel("Task ID", fontsize=18)
            ax.set_xticks(tickmarks)
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
            ax.set_ylabel("Accuracy", fontsize=18)
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            test_stats.update({'Test Accuracies': wandb.Image(acc_fig)})
            # confusion matrix for all tasks combined
            cm_fig = create_cm_fig(
                np.sum(np.array(summary['cmat']), axis=0),
                display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
            )
            test_stats.update({'Confusion All': wandb.Image(cm_fig)})
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current,
                },
                test_stats=test_stats,
            )
        else:
            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Task Accuracies": "\n{:}\n".format(
                        summary['acc']
                    ),
                },
                test_stats=test_stats,
            )
        return summary


def dataio_prep(hparams, csv_path, label_encoder):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
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

        return sig

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
        output_keys=["id", "sig", "class_string_encoded"]
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
    test_datasets = [dataio_prep(hparams, os.path.join(hparams['save_folder'], 'test_task{}_raw.csv'.format(tt)), label_encoder) for tt in range(num_tasks)]
    # this is for linclf evaluation with ideal data case
    if hparams['linclf_train_type'] == 'full':
        for split in ['train', 'valid']:
            prepare_concat_csv(
                [os.path.join(hparams['save_folder'], '{}_task{}_raw.csv'.format(split, tt)) for tt in range(num_tasks)], 'all', hparams['linclf_train_type']
            )
    # all buffers
    replay = {'train': [], 'valid': []}
    cl_acc_table = np.zeros((num_tasks, num_tasks))
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
            # reset linear classifier
            hparams['recoverables']['classifier'] = \
                    hparams['modules']['classifier'] = \
                    hparams['classifier'] = hparams['classifier_fn']()
            # set new checkpointer
            hparams['checkpointer'] = sb.utils.checkpoints.Checkpointer(
                os.path.join(hparams['save_folder'], 'task{}'.format(task_idx)),
                recoverables=hparams['recoverables']
            )
            print('==> Resetting scheduler, counter and linear classifier at {}'.format(hparams['checkpointer'].checkpoints_dir))
        else:
            # reload everything from the interrupted checkpoint
            # set the checkpointer here, and on_fit_start() loads the content
            assert isinstance(hparams['prev_checkpointer'], sb.utils.checkpoints.Checkpointer)
            # initialize epoch counter and lr scheduler for restore
            hparams['recoverables']['lr_scheduler'] = \
                    hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
            hparams['recoverables']['epoch_counter'] = \
                    hparams['epoch_counter'] = hparams['epoch_counter_fn']()
            # reset linear classifier
            hparams['recoverables']['classifier'] = \
                    hparams['modules']['classifier'] = \
                    hparams['classifier'] = hparams['classifier_fn']()
            hparams['checkpointer'] = hparams['prev_checkpointer']
            hparams['checkpointer'].add_recoverables(hparams['recoverables'])
            # TODO: restore any external buffer for data generation here
            if task_idx > 0:
                buffer_cl_acc_table_path = os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx-1),
                    'cl_acc_table.pt'
                )
                if os.path.exists(buffer_cl_acc_table_path):
                    cl_acc_table = torch.load(buffer_cl_acc_table_path)
                replay_path = os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx-1),
                    'replay.pt'
                )
                if os.path.exists(replay_path):
                    replay = torch.load(replay_path)
            print("==> Resuming from interrupted checkpointer at {}".format(hparams['checkpointer'].checkpoints_dir))
            hparams['resume_interrupt'] = False

        # load weights from pretrained embedder and normalizer
        ssl_checkpointer = sb.utils.checkpoints.Checkpointer(
            os.path.join(hparams['ssl_checkpoints_dir'], 'task{}'.format(task_idx)),
            recoverables={
                'embedding_model': hparams['embedding_model'],
                'normalizer': hparams['mean_var_norm'],
            },
        )
        ssl_checkpointer.recover_if_possible(
            min_key='loss',
        )
        for p in hparams['embedding_model'].parameters():
            p.requires_grad = False
        print("==> Recovering embedder checkpointer at {}".format(ssl_checkpointer.checkpoints_dir))

        # TODO: generate task-wise data
        if hparams['linclf_train_type']  == 'buffer':
            curr_train_replay = prepare_task_csv_from_replay(
                os.path.join(hparams['save_folder'], 'train_task{}_raw.csv'.format(task_idx)),
                replay['train'],
                hparams['replay_num_keep'],
            )
            replay['train'] += curr_train_replay
            if hparams['use_mixup']:
                raise ValueError("mixup not allowed in linclf")
                #  train_data = mixup_dataio_prep(
                #      hparams,
                #      os.path.join(hparams['save_folder'], 'train_task{}_raw.csv'.format(task_idx)),
                #      label_encoder,
                #      replay['train'],
                #  )
            else:
                train_data = class_balanced_dataio_prep(
                    hparams,
                    os.path.join(hparams['save_folder'], 'train_task{}_replay.csv'.format(task_idx)),
                    label_encoder,
                )
            curr_valid_replay = prepare_task_csv_from_replay(
                os.path.join(hparams['save_folder'], 'valid_task{}_raw.csv'.format(task_idx)),
                replay['valid'],
                'all',
            )
            replay['valid'] += curr_valid_replay
            valid_data = dataio_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'valid_task{}_replay.csv'.format(task_idx)),
                label_encoder,
            )
        # ideal evaluation with all seen data
        elif hparams['linclf_train_type'] == 'seen':
            for split in ['train', 'valid']:
                prepare_concat_csv(
                    [os.path.join(hparams['save_folder'], '{}_task{}_raw.csv'.format(split, tt)) for tt in range(task_idx+1)], task_idx, hparams['linclf_train_type']
                )
            train_data = dataio_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_task{}_seen.csv'.format(task_idx)),
                label_encoder,
            )
            valid_data = dataio_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'valid_task{}_seen.csv'.format(task_idx)),
                label_encoder,
            )
        # ideal evaluation with all seen and future data
        elif hparams['linclf_train_type'] == 'full':
            train_data = dataio_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_taskall_full.csv'),
                label_encoder,
            )
            valid_data = dataio_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'valid_taskall_full.csv'),
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

        brain = LinearClassifier(
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
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

        # Load the best checkpoint for evaluation
        if num_tasks == 1:
            test_stats = brain.evaluate(
                test_set=test_datasets[0],
                max_key="acc",
                progressbar=True,
                test_loader_kwargs=hparams["valid_dataloader_opts"],
            )

        else:
            # multitask evaluation up to the current task
            if hparams['linclf_train_type'] in ['seen', 'buffer']:
                test_stats = brain.evaluate_multitask(
                    test_datasets[:task_idx+1],
                    max_key='acc',
                    test_loader_kwargs=hparams['valid_dataloader_opts']
                )
                cl_acc_table[task_idx, :task_idx+1] = test_stats['acc']
            elif hparams['linclf_train_type'] == 'full':
                test_stats = brain.evaluate_multitask(
                    test_datasets,
                    max_key='acc',
                    test_loader_kwargs=hparams['valid_dataloader_opts']
                )
                cl_acc_table[task_idx] = test_stats['acc']
            # global buffer
            torch.save(
                test_stats,
                os.path.join(
                    hparams['checkpointer'].checkpoints_dir,
                    'test_stats.pt'
                )
            )
            torch.save(
                cl_acc_table,
                os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx),
                    'cl_acc_table.pt'
                )
            )
            print("\n {} \n".format(cl_acc_table))
            if task_idx == num_tasks - 1:
                print(compute_cl_statistics(cl_acc_table))
            torch.save(
                replay,
                os.path.join(
                    hparams['save_folder'],
                    'task{}'.format(task_idx),
                    'replay.pt'
                )
            )

        hparams['prev_checkpointer'] = hparams['checkpointer']
