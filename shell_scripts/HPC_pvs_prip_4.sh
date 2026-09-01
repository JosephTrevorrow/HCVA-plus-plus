#!/bin/bash

#SBATCH --job-name=ia23938-pvs_prip_4
#SBATCH --output=pvs_prip_4.out
#SBATCH --error=pvs_prip_4.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"
echo Directory is "$(pwd)"

source ~/miniforge3/bin/activate

conda activate abpi

# Ensure Julia is found
#
# "$PATH:$HOME/julia-1.11.7/bin"
export JULIA_BINDIR=$HOME/julia-1.11.7/bin
export PATH=$JULIA_BINDIR:$PATH

echo Julia path set

export JULIA_DEPOT_PATH=$HOME/julia_depot/global
mkdir -p "$JULIA_DEPOT_PATH"

echo Starting Pkg

julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'

echo Added PyCall!

julia -e 'using Pkg; Pkg.add("StatsBase"); Pkg.add("JSON"); Pkg.add("PythonCall"); Pkg.instantiate();'
echo Instantiated!
#julia -e 'include(pwd()* "/abpi_environment/env/action.jl"); using Main.MyActionModule'

julia -e 'using PythonCall; println("PythonCall OK")'

echo Starting Python

# Experiment ``vary_pvs_prip'''
python -O runner.py -min 120 -max 159 -pvs_dir "value_systems/Synthetic/vary_pvs_prip/PVS/" -prip_dir "value_systems/Synthetic/vary_pvs_prip/PriP/" -n_values 4 -n_actions 2 -output_dir "results/SYNTH_vary_pvs_prip/"
