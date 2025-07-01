#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import random
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def parse_utt_id(filename: str):
    name = os.path.splitext(filename)[0]
    match = re.search(r'(?:_|)(\d+)$', name)
    return match.group(1) if match else None

def debug_log(message: str):
    print(f"[DEBUG] {message}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dys-dir", type=str, required=True, help="Directory containing dysarthric wavs")
    parser.add_argument("--healthy-dir", type=str, required=True, help="Directory containing healthy wavs")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for data splits and manifests")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of pairs for train split")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Fraction of pairs for valid split")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of pairs for test split")
    args = parser.parse_args()

    random.seed(1234)
    dys_dir = Path(args.dys_dir)
    healthy_dir = Path(args.healthy_dir)
    out_dir = Path(args.out_dir)

    debug_log(f"Loading files from {dys_dir} and {healthy_dir}")

    # Gather files
    dys_files = list(dys_dir.glob("*.wav"))
    healthy_files = list(healthy_dir.glob("*.wav"))

    if not dys_files:
        debug_log("No dysarthric .wav files found.")
    if not healthy_files:
        debug_log("No healthy .wav files found.")

    # Build dictionaries
    dys_dict, healthy_dict = {}, {}
    for f in dys_files:
        utt_id = parse_utt_id(f.name)
        if utt_id:
            dys_dict.setdefault(utt_id, []).append(f)

    for f in healthy_files:
        utt_id = parse_utt_id(f.name)
        if utt_id:
            healthy_dict.setdefault(utt_id, []).append(f)

    # Match and create pairs
    all_pairs = []
    for utt_id, dys_list in dys_dict.items():
        if utt_id in healthy_dict:
            for d in dys_list:
                for h in healthy_dict[utt_id]:
                    all_pairs.append((utt_id, d, h))

    debug_log(f"Total matched pairs: {len(all_pairs)}")

    if not all_pairs:
        debug_log("No matched pairs found. Check your file naming conventions.")
        return

    # Shuffle and split
    random.shuffle(all_pairs)
    n_total = len(all_pairs)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    n_test = n_total - n_train - n_valid

    train_pairs, valid_pairs, test_pairs = (
        all_pairs[:n_train],
        all_pairs[n_train:n_train + n_valid],
        all_pairs[n_train + n_valid:]
    )

    debug_log(f"Train: {len(train_pairs)}, Valid: {len(valid_pairs)}, Test: {len(test_pairs)}")

    # Create output directories
    src_audio_root = out_dir / "src_audio"
    for split in ["train", "valid", "test"]:
        (src_audio_root / split).mkdir(parents=True, exist_ok=True)

    # Manifest and mapping data
    healthy_manifests = {"train": [], "valid": [], "test": []}
    split_mappings = {"train": [], "valid": [], "test": []}

    sample_counter = 0

    def copy_dys_wav(sample_id, src_path, split):
        out_path = src_audio_root / split / (sample_id + ".wav")
        shutil.copy2(src_path, out_path)
        return out_path

    def write_pair_list(pairs, split):
        nonlocal sample_counter
        for utt_id, dys_path, healthy_path in tqdm(pairs, desc=f"Processing {split}"):
            sample_counter += 1
            sample_id = f"{split}_{sample_counter:06d}"

            # Copy dys wave to output directory
            out_path = copy_dys_wav(sample_id, dys_path, split)
            healthy_abspath = os.path.abspath(healthy_path)

            # Add entries to manifests
            healthy_manifests[split].append(f"{healthy_abspath}\t{sf.info(healthy_path.as_posix()).frames}")
            split_mappings[split].append(f"{sample_id}\t{dys_path.name}\t{healthy_path.name}")

    write_pair_list(train_pairs, "train")
    write_pair_list(valid_pairs, "valid")
    write_pair_list(test_pairs, "test")

    # Write manifest and mapping files
    for split in ["train", "valid", "test"]:
        out_file = out_dir / f"healthy_{split}_quantization_manifest.tsv"
        mapping_file = out_dir / f"{split}_mapping.tsv"

        debug_log(f"Writing {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{healthy_dir.resolve()}\n")
            for line in healthy_manifests[split]:
                f.write(line + "\n")

        debug_log(f"Writing {mapping_file}")
        with open(mapping_file, "w", encoding="utf-8") as f:
            for line in split_mappings[split]:
                f.write(line + "\n")

    debug_log(f"Done! Data splits and mapping files written to {out_dir}")

if __name__ == "__main__":
    main()
