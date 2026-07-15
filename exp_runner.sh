###### ESS DATA EXP. ######

# Experiment ``ESS Country-level'': Run on ESS data, at a country level abstraction (default)

## Data
python -O runner.py -pvs_dir "value_systems/ESS/Country/4_val_2_act/PVS/" -prip_dir "value_systems/ESS/Country/4_val_2_act/PriP/" -n_values 4 -n_actions 2 -e 1e-4 -output_dir "/results/ESS_COUNTRY/4_val_2_acts/"

python -O runner.py -pvs_dir "value_systems/ESS/Country/4_val_3_act/PVS/" -prip_dir "value_systems/ESS/Country/4_val_3_act/PriP/" -n_values 4 -n_actions 3 -e 1e-4 -output_dir "/results/ESS_COUNTRY/4_val_3_acts/"

python -O runner.py -pvs_dir "value_systems/ESS/Country/10_val_2_act/PVS/" -prip_dir "value_systems/ESS/Country/10_val_2_act/PriP/" -n_values 10 -n_actions 2 -e 1e-4 -output_dir "/results/ESS_COUNTRY/10_val_2_acts/"

##### VALE EXPERIMENTS, VALIDATION ##########
python -O runner.py -pvs_dir "value_systems/VALE/PVS/" -prip_dir "value_systems/VALE/PriP/" -n_values 2 -n_actions 1 -e 1e-4 -output_dir "/results/VALE/"

###### SYNTHETIC DATA EXP. ######

# Experiment ``vary_grp_fact'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_grp_fact/PVS/" -prip_dir "value_systems/Synthetic/vary_grp_fact/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_grp_fact/"

# Experiment ``vary_mup_vamu'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_mup_vamu/PVS/" -prip_dir "value_systems/Synthetic/vary_mup_vamu/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_mup_vamu/"

# Experiment ``vary_pvs_prip'''
python -O runner.py -pvs_dir "value_systems/Synthetic/vary_pvs_prip/PVS/" -prip_dir "value_systems/Synthetic/vary_pvs_prip/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_pvs_prip/"

###### ENVIRONMENT VARIABLES, SYNTHETIC DATA EXP. ######

# Experiment 2: Run on VALE data -> Nice for loop!
## Data
#python main.py -f vale_f -pf vale_pf -slm -t -range -hcva -hcva2 -n_values 2 -n_actions 1
## Data with synth principle sets
#for item in $vale_synth_pfs; do
#  python main.py -f vale_f -pf item -slm -hcva -hcva2 -n_values 2 -n_actions 1
#done


