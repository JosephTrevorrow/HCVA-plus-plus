HPC.sh#!/bin/bash

#SBATCH --job-name=ia23938-hcva-2
#SBATCH --output=hcva-2.out
#SBATCH --error=hcva-2.err
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=72

# Some args:
export N_WORKERS=18
export CVXPY_THREADS=$((SLURM_CPUS_PER_TASK / N_WORKERS))

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

echo setting PYTHON_JULIAPKG_EXE

export PYTHON_JULIAPKG_EXE=$HOME/julia-1.11.7/bin/julia
echo "PYTHON_JULIAPKG_EXE=$PYTHON_JULIAPKG_EXE"
echo "PATH=$PATH"
echo Starting Python
python -c "from runner import julia_init; julia_init()" || { echo "Julia pre-warm failed"; exit 1; }
export PYTHON_JULIAPKG_OFFLINE=yes

# Experiment ``vary_grp_fact'''. in 24 hrs, this will only cover vary_grp_fact to some extent.
#srun --export=ALL --ntasks=$N_TASKS --cpus-per-task=$SLURM_CPUS_PER_TASK python -O runner.py -n_workers $N_WORKERS -values_dir "value_systems/Synthetic/vary_grp_fact" "value_systems/Synthetic/vary_mup_vamu" "value_systems/Synthetic/vary_prip_grp_fact" "value_systems/Synthetic/vary_sigma_prip" "value_systems/Synthetic/vary_sigma_prip_MINIMISE" "value_systems/Synthetic/randoms" -n_values 4 -n_actions 2
# All but vary_grp_fact
srun --export=ALL --ntasks=$SLURM_NTASKS --cpus-per-task=$SLURM_CPUS_PER_TASK python -O runner.py -n_workers $N_WORKERS -values_dir "value_systems/Synthetic/vary_mup_vamu" -n_values 4 -n_actions 2

