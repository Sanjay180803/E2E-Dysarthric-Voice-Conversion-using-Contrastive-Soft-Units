import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample

# Load your S2UT model
def load_s2ut_model(model_path):
    print(f"Loading S2UT model from {model_path}")
    model = torch.load(model_path, map_location="cuda")  # Load model
    model.eval()  # Set to evaluation mode
    model.to("cuda")
    return model

def encode_dataset(args):
    # Load your pretrained S2UT model
    s2ut_model = load_s2ut_model(args.model_path)

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        # Load and preprocess audio
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)  # Resample to 16kHz
        wav = wav.unsqueeze(0).cuda()   # Add batch dimension

        with torch.inference_mode():
            # Generate discrete speech units using S2UT model
            units = s2ut_model(wav)  # Assuming your model takes raw audio as input

        # Save as .npy
        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset using S2UT.")
    parser.add_argument(
        "model_path",
        help="Path to the pretrained Speech-to-Unit (S2UT) model.",
        type=Path,
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="Path to the dataset directory containing .wav files.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="Path to the output directory where .npy unit files will be saved.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="Extension of the audio files (defaults to .wav).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
