# Continaul Self-Supervised Learning (CSSL) for Sound Classification

## Requirements
  - pytorch == 1.10.0
  - torchaudio == 0.10.0
  - tqdm == 4.62.3
  - speechbrain == 0.5.11


## Experiments
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


### Linear Evaluation
To run the linear classification on top of the frozen pretrained models, run
```bash
python linclf_train.py hparams/linclf_train.yaml --linclf_train_type buffer --data_folder <YOUR_DATA_FOLDER> --output_base <YOUR_OUTPUT_FOLDER> --ssl_checkpoints_dir <YOUR_SSL_FOLDER>
```
Make sure the `YOUR_SSL_FOLDER` ends with the `save` directory, as the file will redirect under the `save` directory to look for the proper checkpoint for each individual task.

We follow the paradigms in the [Co2L](https://arxiv.org/pdf/2106.14413.pdf), we implement three scenarios for training the linear classifier. By default, we train the linear classifier with all data from the current task along with the replay buffer from previous tasks with the `--linclf_train_type buffer` flag (class-balanced sampling still WIP).

To investigate the tendency for catastrophic forgetting on the algorithms, we pick the embeddings learned without any handling of continual learning (no buffer, finetuning with data only from the current task). Then, we train the linear classifier with the flag `--linclf_train_type seen`, which trains the linear classifier with all data samples from the first to the current task (equivalent to a full buffer).

To evaluate whether the representation learned is useful for future tasks, we train the classifier with the flag `--linclf_train_tyep full`, which trains the classifier with data samples from all tasks on top of the embedder at each task.

