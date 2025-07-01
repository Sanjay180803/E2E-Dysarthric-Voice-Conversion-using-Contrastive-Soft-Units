# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0

# Globals for Mel and Hann Window
mel_basis = {}
hann_window = {}


def get_yaapt_f0(audio, rate=16000, interp=False):
    """Extract F0 using the YAAPT algorithm."""
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        if y.size == 0:
            print(f"[DEBUG] Empty audio array encountered.")
            continue
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        try:
            pitch = pYAAPT.yaapt(
                signal,
                **{
                    "frame_length": frame_length,
                    "frame_space": 5.0,
                    "nccf_thresh1": 0.25,
                    "tda_frame_length": 25.0,
                },
            )
        except Exception as e:
            print(f"[DEBUG] YAAPT failed for this audio: {e}")
            f0s.append(np.zeros_like(y_pad[None, None, :]))
            continue

        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    if len(f0s) == 0:
        print(f"[DEBUG] No F0 values extracted.")
        return np.zeros((1, 1, len(audio[0]) // 80))

    return np.vstack(f0s)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Compute a mel spectrogram from a waveform."""
    if torch.min(y) < -1.:
        print(f"[DEBUG] Min value in input exceeds -1: {torch.min(y)}")
    if torch.max(y) > 1.:
        print(f"[DEBUG] Max value in input exceeds 1: {torch.max(y)}")

    global mel_basis, hann_window

    device_str = "cpu" if not y.is_cuda else str(y.device)
    device_key = f"{fmax}_{device_str}"

    if device_key not in mel_basis:
        mel_filterbank = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[device_key] = torch.from_numpy(mel_filterbank).float().to(y.device)

    if device_str not in hann_window:
        hann_window[device_str] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[device_str],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)
    spec = torch.matmul(mel_basis[device_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_audio(full_path):
    """Load audio using soundfile."""
    try:
        data, sampling_rate = sf.read(full_path, dtype="int16")
    except Exception as e:
        print(f"[DEBUG] Error loading audio file {full_path}: {e}")
        return None, None
    return data, sampling_rate


def spectral_normalize_torch(magnitudes):
    """Normalize spectrogram magnitudes."""
    return torch.log(torch.clamp(magnitudes, min=1e-5))


def parse_manifest(manifest):
    """Parse manifest file for dataset."""
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line.strip():
                sample = eval(line.strip())
                k = next((key for key in ["cpc_km100", "vqvae256", "hubert", "codes"] if key in sample), "codes")
                codes.append(torch.LongTensor([int(x) for x in sample[k].split(" ")]).numpy())
                audio_files.append(Path(sample["audio"]))
    return audio_files, codes


def get_dataset_filelist(h):
    """Get dataset file lists."""
    return parse_manifest(h.input_training_file), parse_manifest(h.input_validation_file)



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C



def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output



def parse_manifest(manifest):
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                elif 'hubert' in sample:
                    k = 'hubert'
                else:
                    k = 'codes'

                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes




def parse_speaker(path, method):
    if type(path) == str:
        path = Path(path)

    if method == 'parent_name':
        return path.parent.name
    elif method == 'parent_parent_name':
        return path.parent.parent.name
    elif method == '_':
        return path.name.split('_')[0]
    elif method == 'single':
        return 'A'
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, f0=None, multispkr=False, pad=None,
                 f0_stats=None, f0_normalize=False, f0_feats=False, f0_median=False,
                 f0_interp=False, vqvae=False):
        self.audio_files, self.codes = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0 = f0
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = None
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        if f0_stats:
            self.f0_stats = torch.load(f0_stats)
        self.multispkr = multispkr
        self.pad = pad
        if self.multispkr:
            spkrs = [parse_speaker(f, self.multispkr) for f in self.audio_files]
            spkrs = list(set(spkrs))
            spkrs.sort()

            self.id_to_spkr = spkrs
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pick interval
        interval_start = 0
        interval_end = max(0, N // lcm - seq_len // lcm)

        start_step = random.randint(interval_start, interval_end) if interval_end > 0 else 0

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = min(v.shape[-1], (start_step + seq_len // lcm) * (lcm // hops[i]))
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        try:
            # Load audio
            if self._cache_ref_count == 0:
                audio, sampling_rate = load_audio(filename)
                if audio is None or sampling_rate != self.sampling_rate:
                    print(f"[DEBUG] Skipping file {filename}: SR mismatch or load failure.")
                    return None

                # Resample if needed
                if sampling_rate != self.sampling_rate:
                    audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

                # Normalize audio
                if self.pad:
                    padding = self.pad - (audio.shape[-1] % self.pad)
                    audio = np.pad(audio, (0, padding), "constant", constant_values=0)
                audio = audio / MAX_WAV_VALUE
                audio = normalize(audio) * 0.95
                self.cached_wav = audio
                self._cache_ref_count = self.n_cache_reuse
            else:
                audio = self.cached_wav
                self._cache_ref_count -= 1

            # Handle codes and audio alignment
            if self.vqvae:
                code_length = audio.shape[0] // self.code_hop_size
            else:
                code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
                code = self.codes[index][:code_length]
            audio = audio[:code_length * self.code_hop_size]

            # Ensure valid audio length
            while audio.shape[0] < self.segment_size:
                audio = np.hstack([audio, audio])
                if not self.vqvae:
                    code = np.hstack([code, code])

            # Convert to tensor
            audio = torch.Tensor(audio).unsqueeze(0)
            if not self.vqvae:
                audio, code = self._sample_interval([audio, code])
            else:
                audio = self._sample_interval([audio])[0]

            # Compute Mel spectrogram
            mel_loss = mel_spectrogram(
                audio, self.n_fft, self.num_mels,
                self.sampling_rate, self.hop_size, self.win_size,
                self.fmin, self.fmax_loss, center=False
            )

            feats = {"code": audio.view(1, -1).numpy() if self.vqvae else code.squeeze()}
            if self.f0:
                try:
                    f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
                except Exception as e:
                    print(f"[DEBUG] F0 extraction failed: {e}")
                    f0 = np.zeros((1, 1, audio.shape[-1] // 80))
                feats['f0'] = f0.squeeze(0)

            # Handle multi-speaker
            if self.multispkr:
                feats['spkr'] = self._get_spkr(index)

            # Normalize F0 if needed
            if self.f0_normalize:
                spkr_id = self._get_spkr(index).item()
                stats = self.f0_stats.get(spkr_id, self.f0_stats.get('default', {}))
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                ii = feats['f0'] != 0
                feats['f0'][ii] = (feats['f0'][ii] - mean) / max(std, 1e-6)  # Avoid division by zero

            return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

        except Exception as e:
            print(f"[DEBUG] Error processing file {filename}: {e}")
            return None

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        return torch.LongTensor([self.spkr_to_id.get(spkr_name, 0)]).view(1).numpy()

    def __len__(self):
        return len(self.audio_files)


