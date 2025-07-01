from pathlib import Path
import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MelDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, discrete: bool = False):
        self.discrete = discrete
        self.mels_dir = root / "mels"
        self.units_dir = root / "discrete" if discrete else root / "soft"

        pattern = "train/*.npy" if train else "dev/*.npy"
        self.metadata = [
            path.relative_to(self.mels_dir).with_suffix("")
            for path in self.mels_dir.rglob(pattern)
        ]

        # Initialize counters for missing files
        self.total_files = len(self.metadata)
        self.missing_files = 0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """Load mel and speech units from disk without forcing a fixed length."""
        path = self.metadata[index]
        mel_path = self.mels_dir / path
        units_path = self.units_dir / path

        try:
            mel = np.load(mel_path.with_suffix(".npy"))  # possibly shape: (128, T) or (T, 128)
            units = np.load(units_path.with_suffix(".npy"))  # shape: [T_units]
        except FileNotFoundError as e:
            self.missing_files += 1
            logger.warning(f"Missing file for {path}: {e}")
            return None  # Skip this file

        # Ensure mel is shape (T, 128), time dimension first
        if mel.shape[0] == 128:
            mel = mel.T  # transpose to (T, 128) if it was (128, T)
        # Convert to tensor
        mel = torch.from_numpy(mel).float()

        # OPTIONAL: Add one extra frame for autoregressive shift
        mel = F.pad(mel, (0, 0, 1, 0))  # now shape is [T_mel+1, 128]

        # Convert units
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()

        return mel, units

    def log_skipped_files(self):
        """If any files were missing, print a summary."""
        if self.missing_files > 0:
            logger.info(
                f"Skipped {self.missing_files}/{self.total_files} files "
                f"({(self.missing_files / self.total_files) * 100:.2f}%) "
                "due to missing data."
            )

    def pad_collate(self, batch):
        """Custom collate function to handle variable-length mels/units."""
        # Filter out None values for missing files
        batch = [item for item in batch if item is not None]

        # If everything was None, return empty
        if not batch:
            return (
                torch.empty(0, 0, 0),
                torch.empty(0),
                torch.empty(0, 0),
                torch.empty(0),
            )

        mels, units = zip(*batch)
        mels, units = list(mels), list(units)

        # Because of the extra pad frame, the "valid" mel length is size(0) - 1
        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        # Pad mels along time dimension
        mels = pad_sequence(mels, batch_first=True)  # shape: [B, max_T, 128]

        # If discrete, pad with ID=1000, else 0
        PAD_VALUE = 1000
        units = pad_sequence(
            units,
            batch_first=True,
            padding_value=PAD_VALUE if self.discrete else 0,
        )

        return mels, mels_lengths, units, units_lengths
