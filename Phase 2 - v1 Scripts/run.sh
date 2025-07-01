#!/bin/bash
#SBATCH --job-name=s2ut_train        # Job name
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --time=0-96:00:00            # Maximum runtime (96 hours)
#SBATCH --partition=cpup             # CPU-only partition
#SBATCH --output=s2ut_train.out      # Output file
#SBATCH --error=s2ut_train.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
DATA_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/s2ut"
MODEL_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/models"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# 3) Start logging all output to the log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.

# 5) Run the training command
python fairseq_cli/train.py "$DATA_ROOT" \
  --config-yaml /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/s2ut/config.yaml \
  --task speech_to_speech \
  --target-is-code \
  --target-code-size 1000 \
  --criterion speech_to_unit \
  --arch s2ut_transformer_fisher \
  --share-decoder-input-output-embed \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --relu-dropout 0.1 \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --clip-norm 10.0 \
  --seed 1 \
  --num-workers 8 \
  --train-subset train \
  --valid-subset valid \
  --save-dir "$MODEL_DIR" \
  --max-tokens 20000 \
  --max-update 400000




