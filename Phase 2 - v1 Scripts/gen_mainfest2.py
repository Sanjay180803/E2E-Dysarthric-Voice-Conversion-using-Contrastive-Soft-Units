import os
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split

# Directory setup
AUDIO_DIR = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data"
HEALTHY_AUDIO_DIR = os.path.join(AUDIO_DIR, "dysarthric")
OUTPUT_DIR = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/manifest3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gather all healthy audio file paths
healthy_audio_paths = []
for root, _, files in os.walk(HEALTHY_AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):  # Adjust extension as needed
            healthy_audio_paths.append(os.path.join(root, file))

# Function to get the number of frames in an audio file
def get_num_frames(audio_path):
    with sf.SoundFile(audio_path) as f:
        return len(f)

# Generate list of (path, num_frames) for each audio file
audio_data = [(os.path.abspath(path), get_num_frames(path)) for path in healthy_audio_paths]

# Split data into train, valid, and test sets (80/10/10 split)
train_data, temp_data = train_test_split(audio_data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Helper function to write manifest files
def write_manifest(data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        # First line is the root path of audio files
        f.write(f"{AUDIO_DIR}\n")
        # Each subsequent line contains <relative_path>\t<num_frames>
        for abs_path, num_frames in data:
            relative_path = os.path.relpath(abs_path, AUDIO_DIR)
            f.write(f"{relative_path}\t{num_frames}\n")

# Write the manifests
write_manifest(train_data, "train.tsv")
write_manifest(valid_data, "valid.tsv")
write_manifest(test_data, "test.tsv")

print("Manifests for healthy speech generated successfully!")
