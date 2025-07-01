# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from pathlib import Path
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


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

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

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

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
        code = self.codes[index][:code_length]
        audio = audio[:code_length * self.code_hop_size]
        assert audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])

        audio = torch.FloatTensor(audio).unsqueeze(0)

        audio, code = self._sample_interval([audio, code])

        feats = {"code": code.squeeze()}
        feats['f0'] = self._extract_f0(audio)

        if self.f0_normalize:
            spkr_id = self._get_spkr(index).item()
            if spkr_id not in self.f0_stats:
                mean = self.f0_stats['mean'].numpy()
                std = self.f0_stats['std'].numpy()
            else:
                mean = self.f0_stats[spkr_id]['mean'].numpy()
                std = self.f0_stats[spkr_id]['std'].numpy()

            ii = feats['f0'] != 0
            if self.f0_median:
                med = np.median(feats['f0'][ii])
                feats['f0'][~ii] = med
            feats['f0'][ii] = (feats['f0'][ii] - mean) / std

            if self.f0_feats:
                feats['f0_stats'] = np.array([mean, std])

        return feats, audio.squeeze(0), str(filename)

    def _extract_f0(self, audio):
        try:
            f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=self.f0_interp)
        except Exception:
            f0 = np.zeros((1, 1, audio.shape[-1] // self.hop_size))
        return f0.squeeze(0)

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def __len__(self):
        return len(self.audio_files)
