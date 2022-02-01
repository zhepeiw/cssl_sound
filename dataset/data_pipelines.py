import speechbrain as sb
import torch
import torchaudio
from scipy.signal import resample_poly
import soundfile as sf
import numpy as np
import pdb


def read_and_resample(path, target_sr, duration=None):
    sig, sr = sf.read(path, dtype='float32', always_2d=True)
    sig = sig.mean(axis=-1)
    if duration is not None:
        seg_len = int(sr * duration)
        if len(sig) < seg_len:
            tmp = np.zeros(seg_len).astype(np.float32)
            rstart = np.random.randint(0, seg_len - len(sig) + 1)
            tmp[rstart:rstart+len(sig)] = sig
            sig = tmp
        elif len(sig) > seg_len:
            start = np.random.randint(0, len(sig) - seg_len + 1)
            sig = sig[start:start+seg_len]
    if sr != target_sr:
        sig = resample_poly(sig, target_sr, sr).astype(np.float32)
    return torch.from_numpy(sig)


def dataio_prep(hparams, csv_path, duration=None):
    config_sample_rate = hparams["sample_rate"]
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path
    )

    @sb.utils.data_pipeline.takes("wav_path", "class_idx", "class_name")
    @sb.utils.data_pipeline.provides("sig", "class", "class_name")
    def audio_pipeline(wav_path, class_idx, class_name):
        class_idx = int(class_idx)
        #  sig, read_sr = torchaudio.load(wav_path)
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
        sig = read_and_resample(wav_path, config_sample_rate, duration)
        #  max_amp = torch.abs(sig).max().item()
        #  if max_amp <= 0:
        #      pdb.set_trace()
        #  scaling = 1 / max_amp * 0.9
        #  sig = scaling * sig
        class_idx = torch.LongTensor([int(class_idx)])
        yield sig
        yield class_idx
        yield class_name

    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_pipeline],
        output_keys=["id", "sig", "class", "class_name"]
    )

    return ds
