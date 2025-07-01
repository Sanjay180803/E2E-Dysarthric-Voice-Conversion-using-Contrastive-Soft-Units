#!/bin/bash
#SBATCH --job-name=sn_train        # Job name
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=20         # Number of CPU cores per task
#SBATCH --time=0-96:00:00          # Maximum runtime (96 hours)
#SBATCH --partition=cpup           # CPU-only partition
#SBATCH --output=sn_train.out      # Output file
#SBATCH --error=sn_train.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
DATA_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/sn"
MODEL_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/sn_models"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr"
LOG_FILE="$LOG_DIR/sntrain_$(date +%Y%m%d_%H%M%S).log"

# 2) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.

# 3) Start logging all output to the log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Train the model using `hubert_ctc`
python fairseq_cli/train.py "$DATA_ROOT" \
  --config-yaml /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/sn/config/config.yaml \
  --multitask-config-yaml /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/sn/config/config_multitask.yaml \
  --task speech_to_text \
  --criterion ctc \
  --arch hubert_ctc \
  --w2v-path /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/kmeans_mhubert_1000.bin \
  --feature-grad-mult 0.0 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0.1 \
  --apply-mask \
  --mask-selection static \
  --mask-length 10 \
  --mask-prob 0.5 \
  --train-subset main_task/train \
  --valid-subset main_task/valid \
  --save-dir "$MODEL_DIR" \
  --lr 5e-5 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --optimizer adam \
  --adam-betas "(0.9,0.98)" \
  --clip-norm 10.0 \
  --max-tokens 20000 \
  --max-update 400000 \
  --update-freq 4 \
  --num-workers 8 \
  --cpu \
  --seed 1
