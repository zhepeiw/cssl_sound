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
    num_keep='all',
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
    if num_keep == 'all':
        curr_buffer = curr_data
    else:
        curr_buffer = rng.choice(
            curr_data,
            min(num_keep, len(curr_data)),
            replace=False
        ).tolist()
    df_agg = pd.DataFrame(curr_data + buffer)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csv.replace('raw', 'replay'), index=False)
    return curr_buffer


def prepare_epmem_csv_from_replay(
    input_csv,
    buffer,
    num_keep='all',
):
    '''
        prepare csv for each task with a rehearsal buffer
        as episodic memory for A-GEM

        args:
            input_csv: str, path to the input csv
            buffer: list of dicts
    '''
    rng = np.random.RandomState(1234)
    df = pd.read_csv(input_csv, index_col=None)
    curr_data = df.to_dict('records')
    if num_keep == 'all':
        curr_buffer = curr_data
    else:
        curr_buffer = rng.choice(
            curr_data,
            min(num_keep, len(curr_data)),
            replace=False
        ).tolist()
    df_agg = pd.DataFrame(buffer)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csv.replace('raw', 'epmem'), index=False)
    return curr_buffer

    
def prepare_task_csv_from_subset(
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
    df_agg.to_csv(input_csv.replace('raw', 'subset'), index=False)
    return curr_buffer


def prepare_concat_csv(
    input_csvs,
    task_idx,
    train_type,
):
    all_data = [pd.read_csv(path, index_col=None).to_dict('records') for path in input_csvs]
    all_data = [item for sublist in all_data for item in sublist]
    df_agg = pd.DataFrame(all_data)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csvs[0].replace('task0_raw', 'task{}_{}'.format(task_idx, train_type)), index=False)


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
        # scaling
        max_amp = torch.abs(sig).max().item()
        #  assert max_amp > 0
        scaling = 1 / max_amp * 0.9
        sig = scaling * sig
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


def mixup_dataio_ssl_prep(
    hparams,
    csv_path,
    label_encoder,
):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    mixup_list = pd.read_csv(csv_path, index_col=None).to_dict('records')

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

    def random_segment(sig, target_len):
        rstart = torch.randint(0, len(sig) - target_len + 1, (1,)).item()
        return sig[rstart:rstart+target_len]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path", "class_name")
    @sb.utils.data_pipeline.provides("sig1", "sig2", "label_prob")
    def mixup_pipeline(wav_path, class_name):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        data_sig = read_sig(wav_path)
        data_string_encoded = label_encoder.encode_label_torch(class_name)
        data_label_onehot = F.one_hot(data_string_encoded, hparams['n_classes']).float()
        mixup_dict = hparams['np_rng'].choice(mixup_list, size=1)[0]
        mixup_sig = read_sig(mixup_dict['wav_path'])
        mixup_name = mixup_dict['class_name']
        mixup_string_encoded = label_encoder.encode_label_torch(mixup_name)
        mixup_label_onehot = F.one_hot(mixup_string_encoded, hparams['n_classes']).float()
        lam = hparams['np_rng'].beta(hparams['mixup_alpha'], hparams['mixup_alpha'])
        min_len = min(len(data_sig), len(mixup_sig))
        sig = data_sig[:min_len] * lam + mixup_sig[:min_len] * (1 - lam)
        assert abs(sig).max() > 0
        target_len = int(hparams["train_duration"] * config_sample_rate)
        if len(sig) > target_len:
            sig1 = random_segment(sig, target_len)
            sig2 = random_segment(sig, target_len)
        else:
            sig1 = sig
            sig2 = sig.clone()
        yield sig1
        yield sig2
        label_prob = data_label_onehot * lam + mixup_label_onehot * (1 - lam)
        yield label_prob

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[mixup_pipeline],
        output_keys=["id", "sig1",  "sig2", "label_prob"]
    )

    return ds


def class_balanced_dataio_prep(
    hparams,
    csv_path,
    label_encoder
):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    class_dict = get_class_dict(pd.read_csv(csv_path, index_col=None))
    class_keys = list(class_dict.keys())

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig", "class_string_encoded", "sample_id")
    def audio_label_pipeline(wav_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        class_name = hparams['np_rng'].choice(class_keys, size=1)[0]
        info = hparams['np_rng'].choice(class_dict[class_name])
        wav_path = info['wav_path']
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
        yield sig
        class_string_encoded = label_encoder.encode_label_torch(class_name)
        yield class_string_encoded
        yield info['ID']


    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_label_pipeline],
        output_keys=["id", "sig", "class_string_encoded", "sample_id"]
    )

    return ds


def get_class_dict(df):
    '''
        returns the mapping from class to data info
    '''
    class2info_dict = {}
    curr_data = df.to_dict('records')
    for info in curr_data:
        class_name = info['class_name']
        if class_name not in class2info_dict:
            class2info_dict[class_name] = []
        class2info_dict[class_name].append(info)
    return class2info_dict

