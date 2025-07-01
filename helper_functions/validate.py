import os
from pydub import AudioSegment

folder = "/nlsasfs/home/nltm-st/sanb/dataset_tdsc_split/dev-*"
for file in os.listdir(folder):
    if file.endswith(".wav"):
        try:
            AudioSegment.from_file(os.path.join(folder, file))
        except Exception as e:
            print(f"Invalid file: {file}, Error: {e}")
