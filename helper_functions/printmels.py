import numpy as np
import glob

all_paths = glob.glob("/nlsasfs/home/nltm-st/sanb/acoustic-model/implementation/dataset-dir/mels/**/*.npy", recursive=True)
for path in all_paths:
    mel = np.load(path)
    print(path, mel.shape)
