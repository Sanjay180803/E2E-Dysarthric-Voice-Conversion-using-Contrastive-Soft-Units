#!/bin/bash
#SBATCH --job-name=hifigan        # Job name
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --time=0-96:00:00            # Maximum runtime (96 hours)
#SBATCH --partition=cpup             # CPU-only partition
#SBATCH --output=hifigan.out      # Output file
#SBATCH --error=hifigan.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate


# 2) Define paths
HIFI_ROOT="/nlsasfs/home/nltm-st/sanb/speech-resynthesis"
CONFIG_PATH="/nlsasfs/home/nltm-st/sanb/speech-resynthesis/examples/speech_to_speech_translation/configs/hubert100_dw1.0.json"
CHECKPOINT_PATH="/nlsasfs/home/nltm-st/sanb/speech-resynthesis/examples/speech_to_speech_translation/checkpoints"

# 3) Start logging all output to the log file and terminal
LOG_DIR="/nlsasfs/home/nltm-st/sanb/speech-resynthesis/examples/speech_to_speech_translation"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$HIFI_ROOT"
export PYTHONPATH=.

# 5) Run the training command
python -m examples.speech_to_speech_translation.train \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH"