import os
import json
import numpy as np
from pathlib import Path
import argparse
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

# Define LogMelSpectrogram class directly in the script
class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        padding = (1024 - 160) // 2
        wav = F.pad(wav, (padding, padding), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel

# Initialize melspectrogram
melspectrogram = LogMelSpectrogram()

def process_wav(in_path, out_path):
    try:
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        logmel = melspectrogram(wav.unsqueeze(0))
        np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())
        return out_path, logmel.shape[-1]
    except Exception as e:
        print(f"Error processing {in_path}: {str(e)}")
        return None, 0

def parse_line(line):
    """Parse a line from train.txt/valid.txt into a dictionary"""
    line = line.strip().replace("'", '"')  # Convert to valid JSON
    return json.loads(line)

def create_directory_structure(base_dir, dataset_type):
    """Create required directory structure"""
    (base_dir / "wavs" / dataset_type).mkdir(parents=True, exist_ok=True)
    (base_dir / "mels" / dataset_type).mkdir(parents=True, exist_ok=True)
    (base_dir / "discrete" / dataset_type).mkdir(parents=True, exist_ok=True)

def process_item(item, base_dir, dataset_type):
    """Process a single data item"""
    try:
        # Create paths
        audio_path = Path(item['audio'])
        rel_path = audio_path.relative_to(audio_path.parent.parent.parent)  # Adjust based on your structure
        stem = rel_path.with_suffix('')

        # Create symbolic links for wavs
        wav_dest = base_dir / "wavs" / dataset_type / rel_path
        wav_dest.parent.mkdir(parents=True, exist_ok=True)
        if not wav_dest.exists():
            os.symlink(audio_path, wav_dest)

        # Save units as numpy array
        units = np.array(list(map(int, item['hubert'].split())), dtype=np.int64)
        units_path = base_dir / "discrete" / dataset_type / stem.with_suffix('.npy')
        units_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(units_path, units)

        # Process and save mel spectrogram
        mel_path = base_dir / "mels" / dataset_type / stem.with_suffix('.npy')
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        process_wav(audio_path, mel_path)
        
        return True
    except Exception as e:
        print(f"Error processing item {item['audio']}: {str(e)}")
        return False

def main(args):
    # Process train and validation files
    for dataset_type, input_file in [('train', args.train_file), ('dev', args.valid_file)]:
        print(f"Processing {dataset_type} data from {input_file}")
        
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Create directory structure
        create_directory_structure(args.dataset_dir, dataset_type)

        # Process items
        success_count = 0
        for line in tqdm(lines, desc=f"Processing {dataset_type} files"):
            item = parse_line(line)
            if process_item(item, args.dataset_dir, dataset_type):
                success_count += 1

        print(f"Successfully processed {success_count}/{len(lines)} {dataset_type} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for acoustic model training')
    parser.add_argument('--dataset_dir', type=Path, required=True,
                      help='Path to output dataset directory')
    parser.add_argument('--train_file', type=Path, required=True,
                      help='Path to train.txt file')
    parser.add_argument('--valid_file', type=Path, required=True,
                      help='Path to valid.txt file')
    
    args = parser.parse_args()
    
    # Create base directory structure
    (args.dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
    (args.dataset_dir / "mels").mkdir(parents=True, exist_ok=True)
    (args.dataset_dir / "discrete").mkdir(parents=True, exist_ok=True)
    
    main(args)