import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torchaudio
from torchaudio.functional import resample

from acoustic.utils import LogMelSpectrogram


melspectrogram = LogMelSpectrogram()


def process_wav(in_path, out_path):
    try:
        print(f"[START] Processing file: {in_path}")
        wav, sr = torchaudio.load(in_path)
        print(f"[INFO] Loaded file: {in_path}, Sample rate: {sr}, Shape: {wav.shape}")
        
        # Resample to 16kHz
        wav = resample(wav, sr, 16000)
        print(f"[INFO] Resampled file: {in_path}, New shape: {wav.shape}")
        
        # Extract log-mel spectrogram
        logmel = melspectrogram(wav.unsqueeze(0))
        print(f"[INFO] Generated log-mel spectrogram for: {in_path}, Shape: {logmel.shape}")
        
        # Save the log-mel spectrogram
        np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())
        print(f"[INFO] Saved spectrogram to: {out_path.with_suffix('.npy')}")
        
        print(f"[END] Finished processing: {in_path}")
        return out_path, logmel.shape[-1]
    except Exception as e:
        print(f"[ERROR] Failed to process file: {in_path}, Error: {e}")
        return None, 0


def preprocess_dataset(args):
    try:
        # Ensure the output directory exists
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output directory created: {args.out_dir}")
        
        print(f"[INFO] Extracting features for directory: {args.in_dir}")
        results = []
        
        # Sequential processing (no parallelism)
        for in_path in tqdm(list(args.in_dir.rglob("*.wav")), desc="Processing files"):
            relative_path = in_path.relative_to(args.in_dir)
            out_path = args.out_dir / relative_path.with_suffix("")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result = process_wav(in_path, out_path)
            results.append(result)
        
        # Summarize results
        lengths = {path.stem: length for path, length in results if path is not None}
        frames = sum(lengths.values())
        frame_shift_ms = 160 / 16000
        hours = frames * frame_shift_ms / 3600
        print(f"[SUMMARY] Wrote {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")
    except Exception as e:
        print(f"[ERROR] Error in preprocessing dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mel-spectrograms for an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    preprocess_dataset(args)
