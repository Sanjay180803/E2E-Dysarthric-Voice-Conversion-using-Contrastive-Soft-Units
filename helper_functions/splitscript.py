import os
import shutil
import random

# Define the input folder containing all .wav files
input_folder = "/nlsasfs/home/nltm-st/sanb/control_combined"
output_folder = "/nlsasfs/home/nltm-st/sanb/dataset_tdsc_split/"

# Define the train and dev directories
train_folder = os.path.join(output_folder, "train-*")
dev_folder = os.path.join(output_folder, "dev-*")

# Create output directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(dev_folder, exist_ok=True)

# Get all .wav files from the input folder
wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

# Shuffle the files to ensure randomness
random.shuffle(wav_files)

# Calculate the split index for 80-20 split
split_index = int(0.8 * len(wav_files))

# Split the files into training and validation sets
train_files = wav_files[:split_index]
dev_files = wav_files[split_index:]

# Helper function to move files to their respective directories
def move_files(file_list, target_folder):
    for file in file_list:
        source = os.path.join(input_folder, file)
        target = os.path.join(target_folder, file)
        shutil.copy(source, target)

# Move training files
move_files(train_files, train_folder)

# Move validation files
move_files(dev_files, dev_folder)

print(f"Dataset split completed!\nTrain folder: {train_folder}\nDev folder: {dev_folder}")
