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

### SimSiam Training
Similar to the settings in the supervised training, to run experiments with SimSiam as a self-supervised method, run

```bash
python simsiam_train.py hparams/simsiam_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_folder <YOUR_OUTPUT_FOLDER>
```
