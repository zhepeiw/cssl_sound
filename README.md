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
python sup_train.py hparams/sup_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_folder <YOUR_OUTPUT_FOLDER>
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
python simsiam_train.py hparams/simsiam_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_folder <YOUR_OUTPUT_FOLDER>
```

To run the linear classification on top of the pretrained models, run
```bash
python linclf_train.py hparams/linclf_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_folder <YOUR_OUTPUT_FOLDER> --ssl_checkpoints_dir <YOUR_SSL_FOLDER>
```
Make sure the `YOUR_SSL_FOLDER` ends with the `save` directory, as the file will redirect under the `save` directory to look for the proper checkpoint for each individual task.
