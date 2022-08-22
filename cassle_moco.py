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

import copy


class SupMOCO(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def init_moco(self, k=512, m=0.99, t=0.3):
        """Initiates queue, key encoder, loss function, key encoder update function, queue update function
        Arguments
        ---------
        k : size of queue
        m : moco momentum for updating key encoder
        t : temperature for contrastive loss
        """

        self.queue_len = k
        self.moco_momentum = m
        self.temperature = t
        self.criterion = torch.nn.CrossEntropyLoss()

        queue = torch.randn(self.hparams.emb_dim, k, requires_grad=False)
        queue = F.normalize(queue, dim=0)

        self.queue = queue.to(self.device)

        self.key_embedding_model = copy.deepcopy(self.modules.embedding_model)
        self.key_projector = copy.deepcopy(self.modules.projector)
        #self.key_predictor = copy.deepcopy(self.modules.predictor)

        for param_k in self.key_embedding_model.parameters():
            param_k.requires_grad = False
        
        for param_k in self.key_projector.parameters():
            param_k.requires_grad = False
        
        """
        for param_k in self.key_predictor.parameters():
            param_k.requires_grad = False
        """
    
    @torch.no_grad()
    def reset_queue(self):
        queue = torch.randn(self.hparams.emb_dim, self.queue_len, requires_grad=False)
        queue = F.normalize(queue, dim=0)

        self.queue = queue.to(self.device)
    
    @torch.no_grad()
    def update_queue(self, keys):
        batch_size = keys.shape[0]
        self.queue = torch.roll(self.queue, -batch_size, 1)
        self.queue[:, -batch_size:].data = (keys.T).data

    @torch.no_grad()
    def key_encoder_update(self):
        for param_q, param_k in zip(self.modules.embedding_model.parameters(), self.key_embedding_model.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1. - self.moco_momentum)
        
        for param_q, param_k in zip(self.modules.projector.parameters(), self.key_projector.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1. - self.moco_momentum)
        """
        for param_q, param_k in zip(self.modules.predictor.parameters(), self.key_predictor.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1. - self.moco_momentum)
        """

    @torch.no_grad()
    def batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs1, lens = batch.sig1
        wavs2, _ = batch.sig2

        x1 = self.prepare_features(wavs1, lens, stage)
        x2 = self.prepare_features(wavs2, lens, stage)
        # Embeddings
        #q = self.modules.predictor(self.modules.projector(self.modules.embedding_model(x1)))
        h = self.modules.embedding_model(x1)
        q = self.modules.projector(h)
        #q = F.normalize(q, dim=2)

        with torch.no_grad():  # no gradient to keys
            self.key_encoder_update()
            #k = self.key_predictor(self.key_projector(self.key_embedding_model(x2)))
            x2_, idx_unshuffle = self.batch_shuffle(x2)
            k = self.key_projector(self.key_embedding_model(x2_))
            #k = F.normalize(k, dim=2)
            k = self.batch_unshuffle(k, idx_unshuffle)

        o = self.modules.classifier(h) # [B, 1, C]

        # return q.squeeze(1), k.squeeze(1), o, lens
    
        if self.hparams.prev_embedding_model is not None:
            self.modules.prev_embedding_model.eval()
            if batch.dist_sig1[0] is not None:
                dist_wavs1, dist_lens = batch.dist_sig1
                dist_wavs2, _ = batch.dist_sig2
                dist_x1 = self.prepare_features(dist_wavs1, dist_lens, stage)
                dist_x2 = self.prepare_features(dist_wavs2, dist_lens, stage)
                e1_prev = self.modules.prev_embedding_model(dist_x1)
                e2_prev = self.modules.prev_embedding_model(dist_x2)
                dist_e1 = self.modules.embedding_model(dist_x1)
                dist_e2 = self.modules.embedding_model(dist_x2)
                e1_hat = self.modules.prev_predictor(dist_e1)
                e2_hat = self.modules.prev_predictor(dist_e2)
            else:
                e1_prev = self.modules.prev_embedding_model(x1)
                # e2_prev = self.modules.prev_embedding_model(x2)
                e1_hat = self.modules.prev_predictor(h)
                # e2_hat = self.modules.prev_predictor(e2)
            return {
                'q': q.squeeze(1),
                'k': k.squeeze(1),
                'o': o,
                'e1_hat': e1_hat,
                'e1_prev': e1_prev.detach(),
                'lens': lens,
            }
        else:
            return {
                'q': q.squeeze(1),
                'k': k.squeeze(1),
                'o': o,
                'lens': lens,
            }

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
        
        
        # vgg
        """
        with torch.cuda.amp.autocast(enabled=False):
            feats = self.modules.compute_features(wavs)  # [B, T, D]
            feats = self.hparams.spec_domain_aug(feats, lens)
        """
        
        # others
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
        # q, k, o, lens = predictions
        
        q, k, o = predictions["q"], predictions["k"], predictions["o"]
        lens = predictions["lens"]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        self.update_queue(k)

        ssl_loss = self.criterion(logits, labels)

        if self.hparams.use_mixup and stage == sb.Stage.TRAIN:
            targets, _ = batch.label_prob
        else:
            targets, _ = batch.class_string_encoded # [bs, 1]
            targets = F.one_hot(targets, o.shape[-1]).float()  # [bs, 1, C]
            
        sup_loss = self.hparams.compute_sup_cost(o, targets, lens)
        
        if self.hparams.prev_embedding_model is not None:
            e1_hat = predictions['e1_hat']
            e1_prev = predictions['e1_prev']
            dist_loss_1 = self.hparams.compute_dist_cost(e1_hat.squeeze(1),
                                                         e1_prev.squeeze(1))
            dist_loss = dist_loss_1
        else:
            dist_loss = torch.zeros(1).to(ssl_loss.device)
            
        loss = self.hparams.sup_weight * sup_loss + self.hparams.ssl_weight * ssl_loss + self.hparams.dist_weight * dist_loss

        loss_dict = {
            'ssl': ssl_loss,
            'sup': sup_loss,
            'dist': dist_loss,
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


def dataio_ssl_prep(hparams, csv_path, label_encoder, dist_list=None):
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

    def read_sig(wav_path):
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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig1", "sig2", "dist_sig1", "dist_sig2")
    def audio_pipeline(wav_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        #  sig, read_sr = torchaudio.load(wav_path)
        #
        #  # If multi-channels, downmix it to a mono channel
        #  sig = torch.squeeze(sig)
        #  if len(sig.shape) > 1:
        #      sig = torch.mean(sig, dim=0)
        #
        #  # Convert sample rate to required config_sample_rate
        #  if read_sr != config_sample_rate:
        #      # Re-initialize sampler if source file sample rate changed compared to last file
        #      if read_sr != hparams["resampler"].orig_freq:
        #          hparams["resampler"] = torchaudio.transforms.Resample(
        #              orig_freq=read_sr, new_freq=config_sample_rate
        #          )
        #      # Resample audio
        #      sig = hparams["resampler"].forward(sig)
        #
        #  # scaling
        #  max_amp = torch.abs(sig).max().item()
        #  #  assert max_amp > 0
        #  scaling = 1 / max_amp * 0.9
        #  sig = scaling * sig

        sig = read_sig(wav_path)

        target_len = int(hparams["train_duration"] * config_sample_rate)
        if len(sig) > target_len:
            sig1 = random_segment(sig, target_len)
            sig2 = random_segment(sig, target_len)
        else:
            sig1 = sig
            sig2 = sig.clone()
        yield sig1
        yield sig2
        # distillation dataset
        if dist_list is None or len(dist_list) == 0:
            yield None
            yield None
        else:
            dist_dict = hparams['np_rng'].choice(dist_list, size=1)[0]
            dist_sig = read_sig(dist_dict['wav_path'])
            if len(dist_sig) > target_len:
                dist_sig1 = random_segment(dist_sig, target_len)
                dist_sig2 = random_segment(dist_sig, target_len)
            else:
                dist_sig1 = dist_sig
                dist_sig2 = dist_sig.clone()
            yield dist_sig1
            yield dist_sig2

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
        output_keys=["id", "sig1", "sig2", "dist_sig1", "dist_sig2", "class_string_encoded"]
    )

    return ds

"""
python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,3,4,5,6,7,8,9,10] --valid_folds=[2] --test_folds=[2] --replay_num_keep=0 --use_mixup False --ssl_weight=1.0 --sup_weight=0.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold2 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,3,4,5,6,7,8,9,10] --valid_folds=[2] --test_folds=[2] --replay_num_keep=0 --use_mixup False --ssl_weight=0.0 --sup_weight=1.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold2 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,3,4,5,6,7,8,9,10] --valid_folds=[2] --test_folds=[2] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+18-03-38_seed_2022+moco9256_shuffled_10_00_fold2/save --emb_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold2_linclf --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,3,4,5,6,7,8,9,10] --valid_folds=[2] --test_folds=[2] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+18-46-38_seed_2022+moco9256_shuffled_00_10_fold2/save --emb_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold2_linclf --num_splits 4

python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,2,3,5,6,7,8,9,10] --valid_folds=[4] --test_folds=[4] --replay_num_keep=0 --use_mixup False --ssl_weight=1.0 --sup_weight=0.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold4 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,2,3,5,6,7,8,9,10] --valid_folds=[4] --test_folds=[4] --replay_num_keep=0 --use_mixup False --ssl_weight=0.0 --sup_weight=1.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold4 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,2,3,4,5,7,8,9,10] --valid_folds=[6] --test_folds=[6] --replay_num_keep=0 --use_mixup False --ssl_weight=1.0 --sup_weight=0.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold6 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python supmoco_train.py hparams/us8k/supmoco_train.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2  --train_folds=[1,2,3,4,5,7,8,9,10] --valid_folds=[6] --test_folds=[6] --replay_num_keep=0 --use_mixup False --ssl_weight=0.0 --sup_weight=1.0 --train_duration=2.0 --emb_norm_type sbn --proj_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold6 --moco_momentum 0.9 --moco_temp 0.3 --moco_dict_size 256 --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,2,3,5,6,7,8,9,10] --valid_folds=[4] --test_folds=[4] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+21-02-25_seed_2022+moco9256_shuffled_10_00_fold4/save --emb_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold4_linclf --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,2,3,5,6,7,8,9,10] --valid_folds=[4] --test_folds=[4] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+21-42-50_seed_2022+moco9256_shuffled_00_10_fold4/save --emb_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold4_linclf --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,2,3,4,5,7,8,9,10] --valid_folds=[6] --test_folds=[6] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+22-25-32_seed_2022+moco9256_shuffled_10_00_fold6/save --emb_norm_type sbn --experiment_name moco9256_shuffled_10_00_fold6_linclf --num_splits 4

python linclf_train.py hparams/us8k/linclf_train_moco.yaml --output_base /home/junkaiwu/outputs/cssl_sound/moco_CL_2 --train_folds=[1,2,3,4,5,7,8,9,10] --valid_folds=[6] --test_folds=[6] --linclf_train_type seen --ssl_checkpoints_dir /home/junkaiwu/outputs/cssl_sound/moco_CL_2/2022-04-01+23-09-51_seed_2022+moco9256_shuffled_00_10_fold6/save --emb_norm_type sbn --experiment_name moco9256_shuffled_00_10_fold6_linclf --num_splits 4
"""


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
        hparams['prepare_split_csv_fn']
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
            if task_idx > 0:
                hparams['recoverables']['prev_predictor'] = \
                        hparams['modules']['prev_predictor'] = \
                        hparams['prev_predictor'] = hparams['prev_predictor_fn']()
                hparams['modules']['prev_embedding_model'] = \
                        hparams['prev_embedding_model'] = hparams['prev_embedding_model_fn']()
                prev_embedding_checkpointer = sb.utils.checkpoints.Checkpointer(
                    hparams['prev_checkpointer'].checkpoints_dir,
                    recoverables={
                        'embedding_model': hparams['prev_embedding_model']
                    },
                )
                prev_embedding_checkpointer.recover_if_possible()
                #  hparams['modules']['prev_embedding_model'] = \
                #          hparams['prev_embedding_model'] = copy.deepcopy(hparams['embedding_model'])
                for p in hparams['prev_embedding_model'].parameters():
                    p.requires_grad = False
            else:
                hparams['prev_predictor'] = None
                hparams['prev_embedding_model'] = None
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
            if task_idx > 0:
                hparams['recoverables']['prev_predictor'] = \
                        hparams['modules']['prev_predictor'] = \
                        hparams['prev_predictor'] = hparams['prev_predictor_fn']()
                hparams['modules']['prev_embedding_model'] = \
                        hparams['prev_embedding_model'] = hparams['prev_embedding_model_fn']()
                prev_embedding_checkpointer = sb.utils.checkpoints.Checkpointer(
                    hparams['prev_checkpointer'].checkpoints_dir,
                    recoverables={
                        'embedding_model': hparams['prev_embedding_model']
                    },
                )
                prev_embedding_checkpointer.recover_if_possible()
                for p in hparams['prev_embedding_model'].parameters():
                    p.requires_grad = False
            else:
                hparams['prev_predictor'] = None
                hparams['prev_embedding_model'] = None
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
        if hparams['use_mixup']:
            train_data = mixup_dataio_ssl_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_task{}_replay.csv'.format(task_idx)),
                label_encoder,
            )
        else:
            if hparams['dist_set'] is None:
                dist_set = None
            elif hparams['dist_set'] == 'buffer':
                dist_set = replay['train']
            else:
                raise ValueError("dist set {} is not supported".format(hparams['dist_set']))
            train_data = dataio_ssl_prep(
                hparams,
                os.path.join(hparams['save_folder'], 'train_task{}_replay.csv'.format(task_idx)),
                label_encoder,
                dist_set,
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
            
        brain = SupMOCO(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        brain.init_moco(k=hparams["moco_dict_size"], m=hparams["moco_momentum"], t=hparams["moco_temp"])

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
