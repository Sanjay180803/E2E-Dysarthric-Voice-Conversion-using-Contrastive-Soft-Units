#!/bin/bash
#SBATCH --job-name=sn_run2        # Job name
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --time=0-96:00:00            # Maximum runtime (96 hours)
#SBATCH --partition=cpup             # CPU-only partition
#SBATCH --output=sn_run2.out      # Output file
#SBATCH --error=sn_run2.err       # Error file

# 1) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
RESULTS_PATH="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/inf_results"
DATA_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/s2ut"
GEN_SUBSET="train"
MODEL_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/inference"

# 3) Start logging all output to the log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.

mkdir -p ${RESULTS_PATH}

python -m fairseq_cli.generate ${DATA_ROOT} \
  --config-yaml ${DATA_ROOT}/config.yaml \
  --task speech_to_speech \
  --target-is-code \
  --target-code-size 1000 \
  --path ${MODEL_DIR}/checkpoint_best.pt \
  --gen-subset ${GEN_SUBSET} \
  --max-tokens 20000 \
  --beam 5 \
  --max-len-a 1 \
  --max-len-b 300 \
  --results-path ${RESULTS_PATH} \
  --num-workers 8 \
  --cpu



