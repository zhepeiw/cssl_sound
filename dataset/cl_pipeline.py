import pandas as pd
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F
import speechbrain as sb
import pdb


def prepare_task_csv_from_replay(
    input_csv,
    buffer,
    num_keep
):
    '''
        prepare csv for each task with a rehearsal buffer

        args:
            input_csv: str, path to the input csv
            output_csv: str, path to the output csv
            buffer: list of dicts
    '''
    rng = np.random.RandomState(1234)
    df = pd.read_csv(input_csv, index_col=None)
    curr_data = df.to_dict('records')
    curr_buffer = rng.choice(
        curr_data,
        min(num_keep, len(curr_data)),
        replace=False
    ).tolist()
    df_agg = pd.DataFrame(curr_data + buffer)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csv.replace('raw', 'replay'), index=False)
    return curr_buffer


def prepare_task_csv_for_linclf(
    input_csv,
    buffer,
    num_keep='all',
):
    '''
        prepare csv for each task with a rehearsal buffer

        args:
            input_csv: str, path to the input csv
            buffer: list of dicts
            num_keep: if 'all', keep all
    '''
    df = pd.read_csv(input_csv, index_col=None)
    curr_data = df.to_dict('records')
    if num_keep == 'all':
        curr_buffer = curr_data
    else:
        rng = np.random.RandomState(1234)
        curr_buffer = rng.choice(
            curr_data,
            min(num_keep, len(curr_data)),
            replace=False
        ).tolist()
    df_agg = pd.DataFrame(curr_buffer + buffer)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csv.replace('raw', 'linclf'), index=False)
    return curr_buffer


def mixup_dataio_prep(
    hparams,
    csv_path,
    label_encoder,
    buffer,
):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

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
        return sig

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path", "class_name")
    @sb.utils.data_pipeline.provides("sig", "label_prob")
    def mixup_pipeline(wav_path, class_name):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        data_sig = read_sig(wav_path)
        data_string_encoded = label_encoder.encode_label_torch(class_name)
        data_label_onehot = F.one_hot(data_string_encoded, hparams['n_classes']).float()
        if len(buffer) > 0:
            buffer_dict = hparams['np_rng'].choice(buffer, size=1)[0]
            buffer_sig = read_sig(buffer_dict['wav_path'])
            buffer_name = buffer_dict['class_name']
            buffer_string_encoded = label_encoder.encode_label_torch(buffer_name)
            buffer_label_onehot = F.one_hot(buffer_string_encoded, hparams['n_classes']).float()
            lam = hparams['np_rng'].beta(hparams['mixup_alpha'], hparams['mixup_alpha'])
            min_len = min(len(data_sig), len(buffer_sig))
            sig = data_sig[:min_len] * lam + buffer_sig[:min_len] * (1 - lam)
            assert abs(sig).max() > 0
            yield sig
            label_prob = data_label_onehot * lam + buffer_label_onehot * (1 - lam)
            yield label_prob
        else:
            yield data_sig
            yield data_label_onehot

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[mixup_pipeline],
        output_keys=["id", "sig", "label_prob"]
    )

    return ds
