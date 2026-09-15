#!/bin/bash

#SBATCH --job-name=ia23938-hcva
#SBATCH --output=hcva.out
#SBATCH --error=hcva.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --nodes=3
#SBATCH --cpus-per-task=384

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"
echo Directory is "$(pwd)"

source ~/miniforge3/bin/activate

conda activate hcva

# Ensure Julia is found
#
# "$PATH:$HOME/julia-1.11.7/bin"
export JULIA_BINDIR=$HOME/julia-1.11.7/bin
export PATH=$JULIA_BINDIR:$PATH

echo Julia path set

export JULIA_DEPOT_PATH=$HOME/julia_depot/global
mkdir -p "$JULIA_DEPOT_PATH"

echo Starting Python

# Experiment ``vary_grp_fact'''
python -O runner.py -values_dir "value_systems/Synthetic/vary_grp_fact" "value_systems/Synthetic/vary_mup_vamu" "value_systems/Synthetic/vary_prip_grp_fact" "value_systems/Synthetic/vary_pvs_prip" "value_systems/Synthetic/vary_sigma_prip_MINIMISE" "value_systems/Synthetic/randoms" -n_values 4 -n_actions 2 -n_workers 24
