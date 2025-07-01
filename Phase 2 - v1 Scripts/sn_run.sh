#!/bin/bash
#SBATCH --job-name=sn_run        # Job name
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --time=0-96:00:00            # Maximum runtime (96 hours)
#SBATCH --partition=cpup             # CPU-only partition
#SBATCH --output=sn_run.out      # Output file
#SBATCH --error=sn_run.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# 3) Start logging all output to the log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.

python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/src_audio \
  --ext wav \
  --data-name train \
  --output-dir /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/inference \
  --for-inference


