#!/bin/bash
#SBATCH --job-name=cluster               # Job name
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=20             # Number of CPU cores per task
#SBATCH --time=0-96:00:00              # Maximum runtime (96 hours)
#SBATCH --partition=cpup               # CPU-only partition
#SBATCH --output=cluster.out              # Output file
#SBATCH --error=cluster.err               # Error file

# 2) Activate your conda/virtual environment
source /nlsasfs/home/nltm-st/sanb/fs_env/bin/activate

# 2) Define paths and variables
FAIRSEQ_ROOT="/nlsasfs/home/nltm-st/sanb/fairseq"
LOG_DIR="/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr"
LOG_FILE="$LOG_DIR/cluster_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 4) Navigate to the Fairseq directory
cd "$FAIRSEQ_ROOT"
export PYTHONPATH=.

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file /nlsasfs/home/nltm-st/sanb/acoustic-model/s2ut_results_cpu/generate-valid.unit \
  --vocoder /nlsasfs/home/nltm-st/sanb/acoustic-model/g_00500000 \
  --vocoder-cfg /nlsasfs/home/nltm-st/sanb/acoustic-model/config.json \
  --results-path /nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/final_synthesis \
  --dur-prediction


