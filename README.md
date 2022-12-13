# Continaul Self-Supervised Learning (CSSL) for Sound Classification
This is the code base for the paper ["Learning Representations for New Sound Classes With Continual Self-Supervised Learning"](https://arxiv.org/abs/2205.07390). This paper is accepted by the IEEE Signal Processing Letters.

## Requirements
  - pytorch == 1.10.0
  - torchaudio == 0.10.0
  - tqdm == 4.62.3
  - speechbrain == 0.5.11
  - torchlibrosa == 0.0.9
  - torchmetrics == 0.5.1


## Experiments
### UrbanSound8K
We produce the scripts for reproducing the supervised and self-supervised experiments for the class-incremental representation learning. The following command will run the encoder training using the simclr objective on the UrbanSound8K dataset:

```bash
python supclr_train.py hparams/us8k/supclr_train.yaml --output_base <ENCODER_OUTPUT_DIR> --replay_num_keep=0 --use_mixup=False --ssl_weight=1 --sup_weight=0 --train_duration=2.0 --emb_norm_type=bn --proj_norm_type=bn --batch_size=32 --experiment_name=<NAME>
```

And all configurations can be modified at `./hparams/us8k/supclr_train.yaml`. Make sure to configure the path to the datasets properly. After the encoder is trained, use the linear evaluation script to measure the performance of the learned representations:

```bash
python linclf_train.py hparams/us8k/linclf_train.yaml --output_base <EVAL_OUTPUT_DIR> --linclf_train_type=seen --replay_num_keep=all --ssl_checkpoints_dir=<PATH_TO_ENCODER_CKPT> --emb_norm_type bn --experiment_name=<NAME>
```
Here, the `<PATH_TO_ENCODER_CKPT>` should be identical to the `<ENCODER_OUTPUT_DIR>` with a `/save` suffix. The script will look for the task-wise encoder checkpoint under this directory when evaluating on each task. The `linclf_train_type` is set as `seen`, which will perform linear evaluation protocol (LEP). It can be assigned a value of `subset`, which will instead perform subset linear evaluation protocol (SLEP).

### DCASE TAU19
For encoder training, run
```bash
python supclr_train.py hparams/tau19/supclr_train.yaml --output_base <ENCODER_OUTPUT_DIR> --replay_num_keep=0 --use_mixup=False --ssl_weight=1 --sup_weight=0 --train_duration=4.0 --emb_norm_type bn --proj_norm_type bn --batch_size=32 --experiment_name=<NAME>
```

For in-domain linear evaluation, run
```bash
python linclf_train.py hparams/tau19/linclf_train.yaml --output_base <EVAL_OUTPUT_DIR> --linclf_train_type=seen --ssl_checkpoints_dir=<PATH_TO_ENCODER_CKPT> --emb_norm_type bn --experiment_name=<NAME>
```

### VGGSound
For encoder training, run
```bash
python supclr_train.py hparams/vgg/supclr_train.yaml --output_base <ENCODER_OUTPUT_DIR> --replay_num_keep=0 --use_mixup=False --ssl_weight=1 --sup_weight=0 --train_duration=4.0 --emb_norm_type bn --proj_norm_type bn --batch_size=32 --experiment_name=<NAME>
```

## (Deprecated) Sample Commands
The following section may be outdated, and please refer to the Experiments section.
### Supervised Training

To run the supervised class-incremental experiment, run

```bash
python sup_train.py hparams/sup_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_base <YOUR_OUTPUT_FOLDER>
```

To run the supervised offline training (full dataset at once), replace the arguments of `task_classes` in `hparams/sup_train.yaml` with

```yaml
task_classes:
    - !tuple (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
```
before running the training script.


You may want to checkout the following configurations in the hyperparameter yaml file to configure the experiments:
  - `replay_num_keep`: size of the replay buffer for each task
  - `use_mixup`: flag of using mixup as a data augmentation
  - `training_folds`, `valid_folds`, `test_folds`: urbansound8k fold index for training, validation and test
  - `embedding_model`: encoder/backbone model, currently support the TDNN and PANN architectures

### SimSiam Training
Similar to the settings in the supervised training, to run experiments with SimSiam as a self-supervised method, run

```bash
python simsiam_train.py hparams/simsiam_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_base <YOUR_OUTPUT_FOLDER>
```

### SimCLR Training
Similar to the settings in the supervised and SimSiam training, to run experiments with SimCLR as a self-supervised method, run

```bash
python simclr_train.py hparams/simclr_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_base <YOUR_OUTPUT_FOLDER>
```

### Linear Evaluation
To run the linear classification on top of the frozen pretrained models, run
```bash
python linclf_train.py hparams/linclf_train.yaml --linclf_train_type buffer --data_folder <YOUR_DATA_FOLDER> --output_base <YOUR_OUTPUT_FOLDER> --ssl_checkpoints_dir <YOUR_SSL_FOLDER>
```
Make sure the `YOUR_SSL_FOLDER` ends with the `save` directory, as the file will redirect under the `save` directory to look for the proper checkpoint for each individual task.

We follow the paradigms in the [Co2L](https://arxiv.org/pdf/2106.14413.pdf), we implement three scenarios for training the linear classifier. By default, we train the linear classifier with all data from the current task along with the replay buffer from previous tasks with the `--linclf_train_type buffer` flag (class-balanced sampling still WIP).

To investigate the tendency for catastrophic forgetting on the algorithms, we pick the embeddings learned without any handling of continual learning (no buffer, finetuning with data only from the current task). Then, we train the linear classifier with the flag `--linclf_train_type seen`, which trains the linear classifier with all data samples from the first to the current task (equivalent to a full buffer).

To evaluate whether the representation learned is useful for future tasks, we train the classifier with the flag `--linclf_train_tyep full`, which trains the classifier with data samples from all tasks on top of the embedder at each task.


### Visualization
We provide a notebook for visualizing the learned embeddings for continually trained embedding models. Please configure the following in `hparams/visualize.yaml`:

  - `csv_path`: this is for loading the training and test data, and we recommend loading the offline dataset with all tasks
  - `ssl_checkpoints_dir`: same as in linear evaluation
