import numpy as np
import glob

mels_dir = "/nlsasfs/home/nltm-st/sanb/acoustic-model/implementation/dataset-dir/mels/train"
units_dir = "/nlsasfs/home/nltm-st/sanb/acoustic-model/implementation/dataset-dir/discrete/train"

ratios = []
for mel_path in glob.glob(mels_dir + "/*.npy"):
    # The corresponding units file should have the same stem
    # e.g. /path/to/discrete/train/FC01_327.npy
    filename = mel_path.split("/")[-1]
    units_path = units_dir + "/" + filename  # same stem
    
    mel = np.load(mel_path)
    # mel might be shape (128, T) or (T, 128). We want time dimension T:
    # If your dataset is transposed, then T = mel.shape[0].
    # If not transposed, T = mel.shape[1].
    M = mel.shape[-1]
    
    units = np.load(units_path)
    U = len(units)

    ratio = M / float(U) if U != 0 else 0
    ratios.append(ratio)
    print(filename, U, M, f"ratio={ratio:.2f}")

# Then check the distribution of ratios:
import statistics
print("Mean ratio:", statistics.mean(ratios))
print("Std dev ratio:", statistics.pstdev(ratios))
