#!/bin/bash
#SBATCH --job-name=BayesSupplyArray
#SBATCH --cpus-per-task=32     # 4 chains * 8 threads = 32 CPUs
#SBATCH --mem=64G              # Increased memory from 16G to 64G for stability
#SBATCH --time=48:00:00        # Added 48 hour time limit to prevent premature termination
#SBATCH --output=logs/supply_model_%a_%j.out
#SBATCH --error=logs/supply_model_%a_%j.err
#SBATCH --array=1-4            # Adjusted to 1-4 (Bertrand, Auction, Cournot, MonCom)

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
mkdir -p logs

# Run the R script
# Args: [model_id] [chains] [threads] [use_cutoff]
srun Rscript code/bayesian.R ${SLURM_ARRAY_TASK_ID} $STAN_NUM_CHAINS $STAN_NUM_THREADS 0

# Quick status check after run
if [ $? -eq 0 ]; then
    echo "Model ${SLURM_ARRAY_TASK_ID} completed successfully at $(date)"
else
    echo "Model ${SLURM_ARRAY_TASK_ID} failed with exit code $? at $(date)"
fi
