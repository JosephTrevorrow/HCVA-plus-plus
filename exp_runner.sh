#!/bin/bash

#SBATCH --job-name=ia23938-exp_run
#SBATCH --output=exp.out
#SBATCH --error=exp.err
#SBATCH --time=0:15:00
#SBATCH --mem=32G

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"
echo Directory is "$(pwd)"

for i in {1..5}; do
  echo "Iteration $i"
  sbatch shell_scripts/HPC_grp_fact_$i.sh
  sbatch shell_scripts/HPC_mup_vamu_$i.sh
  sbatch shell_scripts/HPC_pvs_prip_$i.sh
  sbatch shell_scripts/HPC_prip_grp_fact_$i.sh
  sbatch shell_scripts/HPC_pvs_prip_minimise_$i.sh
  sbatch shell_scripts/HPC_random_$i.sh
done
