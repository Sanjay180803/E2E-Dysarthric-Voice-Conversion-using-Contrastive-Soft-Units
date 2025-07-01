#!/bin/bash
#SBATCH --job-name=s2ut_infer        # Job name
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --time=0-96:00:00            # Maximum runtime (96 hours)
#SBATCH --partition=cpup             # CPU-only partition
#SBATCH --output=s2ut_infer.out      # Output file
#SBATCH --error=s2ut_infer.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
DATA_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/s2ut"
MODEL_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/models"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/acoustic-model"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# 3) Start logging all output to the log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.


RESULTS_PATH="/nlsasfs/home/nltm-st/sanb/acoustic-model/s2ut_results_cpu"

# 3) Create output directory
mkdir -p ${RESULTS_PATH}

# 4) CPU-specific inference command
python -m fairseq_cli.generate ${DATA_ROOT} \
  --config-yaml ${DATA_ROOT}/config.yaml \
  --task speech_to_speech \
  --target-is-code \
  --target-code-size 1000 \
  --path ${MODEL_DIR}/checkpoint_best.pt \
  --gen-subset "train" \
  --max-tokens 20000 \
  --beam 5 \
  --max-len-a 1 \
  --max-len-b 300 \
  --results-path ${RESULTS_PATH} \
  --num-workers 8 \
  --cpu 