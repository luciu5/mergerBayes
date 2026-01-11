#!/bin/bash
#SBATCH --job-name=BayesSupplyArray
#SBATCH --cpus-per-task=32     # Increased from 4 to 16 for parallel threads
#SBATCH --mem=16G              # Increased memory to support parallelization
#SBATCH --output=supply_model_%a.out
#SBATCH --array=1-4

# Set number of chains and threads per chain
export STAN_NUM_CHAINS=4      # Number of chains to run in parallel
export STAN_NUM_THREADS=8     # Threads per chain

# Avoid thread oversubscription by BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1

# Set threading environment variables
export TBB_CXX_TYPE=gcc
export MKL_NUM_THREADS=$STAN_NUM_THREADS
export OMP_NUM_THREADS=$STAN_NUM_THREADS

# Run the R script with model number and thread count
srun Rscript ${HOME}/supreme/code/bayes/bayesian.R ${SLURM_ARRAY_TASK_ID} $STAN_NUM_CHAINS  $STAN_NUM_THREADS
