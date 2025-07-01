import os
import shutil

root = "/nlsasfs/home/nltm-st/sanb/acoustic-model/implementation/dataset-dir"

# Subfolders you need to fix
subfolders = ["discrete", "mels", "wavs"]
# Splits that have 'data/healthy' subdirs
splits = ["train", "dev"]

for subfolder in subfolders:
    for split in splits:
        # Path to the "data/healthy" folder
        src_dir = os.path.join(root, subfolder, split, "data", "healthy")
        if not os.path.isdir(src_dir):
            continue  # skip if this doesn't exist

        # Destination is just "train" or "dev"
        dst_dir = os.path.join(root, subfolder, split)

        # Move (rename) each file from "data/healthy" to "train" or "dev"
        for item in os.listdir(src_dir):
            src_path = os.path.join(src_dir, item)
            dst_path = os.path.join(dst_dir, item)
            os.rename(src_path, dst_path)

        # Remove the now-empty "data/healthy" and "data" folders
        # (os.removedirs removes intermediate empty directories too)
        os.removedirs(src_dir)

print("Done! Folder structure flattened.")
