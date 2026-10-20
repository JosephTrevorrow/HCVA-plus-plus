#!/bin/bash

#SBATCH --job-name=ia23938-hcva-da
#SBATCH --output=hcva-da.out
#SBATCH --error=hcva-da.err
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --nodes=1

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"
echo Directory is "$(pwd)"

source ~/miniforge3/bin/activate

conda activate hcva
value_systems/ESS//PVS/

###### ESS DATA EXP. ######

# Experiment ``ESS Country-level'': Run on ESS data, at a country level abstraction (default)
python data_analysis/data_analysis_main.py -ess -cons_dir "/results/Country" -agents_pvs_dir "/value_systems/ESS/Country/PVS/" -agents_prip_dir "/value_systems/ESS/Country/PriP/" -output_dir "/plots/ESS/"

#python data_analysis/data_analysis_main.py -single_timestep_plots -cons_dir "/results/ESS_COUNTRY/4_val_3_acts/" -agents_pvs_dir "/value_systems/ESS/Country/4_val_3_act/PVS/" -agents_prip_dir "/value_systems/ESS/Country/4_val_3_act/PriP/" -output_dir "/ESS_4_val_3_act/"

#python data_analysis/data_analysis_main.py -single_timestep_plots -cons_dir "/results/ESS_COUNTRY/10_val_2_acts/" -agents_pvs_dir "/value_systems/ESS/Country/10_val_2_act/PVS/" -agents_prip_dir "/value_systems/ESS/Country/10_val_2_act/PriP/" -output_dir "/ESS_10_val_2_act/"

###### SYNTHETIC DATA EXP. 500 RUNS######

# -steps is -1 the number of actual files per exp, because we start counting from 0

# Experiment ``vary_grp_fact'''
#python data_analysis/data_analysis_main.py -synth -steps 9 -cons_dir "results/vary_grp_fact/" -agents_pvs_dir "value_systems/Synthetic/vary_grp_fact/PVS/" -agents_prip_dir "value_systems/Synthetic/vary_grp_fact/PriP/" -output_dir "plots/vary_grp_fact_oct/"

# Experiment ``vary_mup_vamu'''
#python data_analysis/data_analysis_main.py -synth -steps 9 -cons_dir "results/vary_mup_vamu/" -agents_pvs_dir "value_systems/Synthetic/vary_mup_vamu/PVS/" -agents_prip_dir "value_systems/Synthetic/vary_mup_vamu/PriP/" -output_dir "plots/vary_mup_vamu_oct/"

#python data_analysis/data_analysis_main.py -synth -steps 8 -cons_dir "results/vary_sigma_prip/" -agents_pvs_dir "value_systems/Synthetic/vary_sigma_prip/PVS/" -agents_prip_dir "value_systems/Synthetic/vary_sigma_prip/PriP/" -output_dir "plots/vary_sigma_prip_oct/"

#python data_analysis/data_analysis_main.py -synth -steps 8 -cons_dir "results/vary_sigma_prip_MINIMISE/" -agents_pvs_dir "value_systems/Synthetic/vary_sigma_prip_MINIMISE/PVS/" -agents_prip_dir "value_systems/Synthetic/vary_sigma_prip_MINIMISE/PriP/" -output_dir "plots/vary_sigma_prip_MINIMISE_oct/"

#python data_analysis/data_analysis_main.py -synth -steps 8 -cons_dir "results/vary_prip_grp_fact/" -agents_pvs_dir "value_systems/Synthetic/vary_prip_grp_fact/PVS/" -agents_prip_dir "value_systems/Synthetic/vary_prip_grp_fact/PriP/" -output_dir "plots/vary_prip_grp_fact_oct/"

#python data_analysis/data_analysis_main.py -synth -steps 0 -cons_dir "results/randoms/" -agents_pvs_dir "value_systems/Synthetic/randoms/PVS/" -agents_prip_dir "value_systems/Synthetic/randoms/PriP/" -output_dir "plots/randoms_oct/"
