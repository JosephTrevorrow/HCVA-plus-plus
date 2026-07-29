#!/bin/bash

#SBATCH --job-name=ia23938-Synth
#SBATCH --output=Synth.out
#SBATCH --error=Synth.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Starting Python

###### SYNTHETIC DATA EXP. ######

# Experiment ``vary_grp_fact'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_grp_fact/PVS/" -prip_dir "value_systems/Synthetic/vary_grp_fact/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_grp_fact/"

# Experiment ``vary_mup_vamu'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_mup_vamu/PVS/" -prip_dir "value_systems/Synthetic/vary_mup_vamu/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_mup_vamu/"

# Experiment ``vary_pvs_prip'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_pvs_prip/PVS/" -prip_dir "value_systems/Synthetic/vary_pvs_prip/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_pvs_prip/"
