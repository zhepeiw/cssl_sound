# Continaul Self-Supervised Learning (CSSL) for Sound Classification

## Requirements
  - pytorch == 1.10.0
  - torchaudio == 0.10.0
  - tqdm == 4.62.3
  - speechbrain == 0.5.11


## Experiments
### Supervised Training

To run the supervised class-incremental experiment, run
```python
python sup_train.py hparams/sup_train.yaml --data_folder <YOUR_DATA_FOLDER> --output_folder <YOUR_OUTPUT_FOLDER>
```

To run the supervised offline training (full dataset at once), replace the arguments of `task_classes` in `hparams/sup_train.yaml` with


```yaml
task_classes:
    - !tuple (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
```
before running the training script.
