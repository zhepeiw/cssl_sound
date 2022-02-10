import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import torch
import torchaudio

import pdb


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


def compare_similarity(x1, x2):
    criterion = torch.nn.CosineSimilarity(dim=2)
    hparams['embedding_model'].eval()
    with torch.no_grad():
        feats1 = compute_features(x1)
        z1 = hparams['embedding_model'](feats1)
        feats2 = compute_features(x2)
        z2 = hparams['embedding_model'](feats2)
        sim = criterion(z1, z2)
    return sim

def compute_features(wavs):
    lens = torch.ones(wavs.shape[0]).to(wavs.device)
    wavs_aug = hparams['time_domain_aug'](wavs, lens)
    if wavs_aug.shape[1] > wavs.shape[1]:
        wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
    else:
        zero_sig = torch.zeros_like(wavs)
        zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
        wavs_aug = zero_sig
    wavs = wavs_aug
    feats = hparams['compute_features'](wavs)
    feats = hparams['mean_var_norm'](feats, lens)
    return feats

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # setting up experiment stamp

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        # load weights from pretrained embedder and normalizer

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(hparams['label_encoder_path'])
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    task_idx = 0
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

    ds = dataio_prep(
        hparams,
        '/mnt/data/zhepei/outputs/cssl_sound/results/2022-02-05+06-01-38_seed_1234+offline/save/train_task0_replay.csv',
        label_encoder
    )
    for _ in range(20):
        idx = torch.randint(len(ds), (1,)).item()
        x1 = ds[idx]['sig'].unsqueeze(0)
        sim = compare_similarity(x1, x1)
        print('ID {} class {} sim={}'.format(
            idx,
            ds[idx]['class_string_encoded'].item(),
            sim.item()
        ))
    for _ in range(20):
        idx1, idx2 = torch.randint(len(ds), (2,))
        x1 = ds[idx1]['sig'].unsqueeze(0)
        x2 = ds[idx2]['sig'].unsqueeze(0)
        sim = compare_similarity(x1, x2)
        print('ID {} class {} - ID {} {} sim={}'.format(
            idx1,
            ds[idx1]['class_string_encoded'].item(),
            idx2,
            ds[idx2]['class_string_encoded'].item(),
            sim.item()
        ))
