#!/bin/bash

#SBATCH --job-name=ia23938-hcva
#SBATCH --output=hcva.out
#SBATCH --error=hcva.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=72

# An hour into computation mem use is sitting at around 8 gig, so can request much less than 100G

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

# These lines should do same thing:
#julia -e 'using Pkg; Pkg.activate("/lfs1i3/home/b35t/jtrevorrow.b35t/miniforge3/envs/hcva/julia_env/Project.toml"); include("lp_regression/IRLS-pNorm.jl")'
python -c "from runner import _worker_init; _worker_init()"

echo setting PYTHON_JULIAPKG_EXE

export PYTHON_JULIAPKG_EXE=$HOME/julia-1.11.7/bin/julia

echo Starting Python

# Experiment ``vary_grp_fact'''
python -O runner.py -values_dir "value_systems/Synthetic/vary_grp_fact" "value_systems/Synthetic/vary_mup_vamu" "value_systems/Synthetic/vary_prip_grp_fact" "value_systems/Synthetic/vary_sigma_prip" "value_systems/Synthetic/vary_sigma_prip_MINIMISE" "value_systems/Synthetic/randoms" -n_values 4 -n_actions 2 -n_workers 42
