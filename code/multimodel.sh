#!/bin/bash
#SBATCH --job-name=BayesSupplyArray
#SBATCH --cpus-per-task=32     # 4 chains * 8 threads = 32 CPUs
#SBATCH --mem=64G              # Increased memory from 16G to 64G for stability
#SBATCH --time=48:00:00        # Added 48 hour time limit to prevent premature termination
#SBATCH --output=logs/model_%a_%j.log   # Combined stdout+stderr to single log
#SBATCH --error=logs/model_%a_%j.log    # Same file for both streams
#SBATCH --array=1-4            # Adjusted to 1-4 (Bertrand, Auction, Cournot, MonCom)

# Model names for readable output
MODEL_NAMES=("" "bertrand" "auction" "cournot" "moncom")
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

# Set number of chains and threads per chain
export STAN_NUM_CHAINS=4      
export STAN_NUM_THREADS=8     

# Avoid thread oversubscription by BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1

# Stan specific threading
export TBB_CXX_TYPE=gcc
export STAN_NUM_THREADS=$STAN_NUM_THREADS

# Create logs directory
LOGDIR="${HOME}/Projects/mergerBayes/logs"
mkdir -p "$LOGDIR"

LOGFILE="${LOGDIR}/model_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log"

# Print job header
echo "=== Job ${SLURM_JOB_ID} | Model ${SLURM_ARRAY_TASK_ID} (${MODEL_NAME}) | Started $(date) ==="

# Run the R script, capturing exit code
# Args: [model_id] [chains] [threads] [use_cutoff] [use_hmt] [data_frac] [filter_year]
# Running for Year 2014 (High Variation) with 100% of data
srun Rscript ${HOME}/Projects/mergerBayes/code/bayesian.R ${SLURM_ARRAY_TASK_ID} $STAN_NUM_CHAINS $STAN_NUM_THREADS 0 0 1.0 2014
EXIT_CODE=$?

# Status check after run
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== SUCCESS: Model ${MODEL_NAME} completed at $(date) ==="
else
    echo "=== FAILED: Model ${MODEL_NAME} (Job ${SLURM_JOB_ID}) exited with code ${EXIT_CODE} at $(date) ==="
    # Rename to highlight failure
    mv "${LOGFILE}" "${LOGDIR}/FAILED_${MODEL_NAME}_${SLURM_JOB_ID}.log" 2>/dev/null
fi

exit $EXIT_CODE

