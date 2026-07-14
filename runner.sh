###### ESS DATA EXP. ######

# Experiment ``ESS Country-level'': Run on ESS data, at a country level abstraction (default)

## Data
#python -O runner.py -pvs_dir "value_systems/ESS/Country/4_val_2_act/PVS/" -prip_dir "value_systems/ESS/Country/4_val_2_act/PriP/" -n_values 4 -n_actions 2 -e 1e-4 -output_dir "/results/ESS_COUNTRY/4_val_2_acts/"
## Data Analysis
#python data_analysis/data_analysis_main.py -cons_dir "results/ESS_COUNTRY/4_val_2_act/" -agents_pvs_dir "value_systems/ESS/Country/4_val_2_act/PVS/" -agents_prip_dir "value_systems/ESS/Country/4_val_2_act/PriP/" -output_dir "plots/ESS_4_val_2_act/"

#python -O runner.py -pvs_dir "value_systems/ESS/Country/4_val_3_act/PVS/" -prip_dir "value_systems/ESS/Country/4_val_3_act/PriP/" -n_values 4 -n_actions 3 -e 1e-4 -output_dir "/results/ESS_COUNTRY/4_val_3_acts/"
## TODO: Insert data analysis

#python -O runner.py -pvs_dir "value_systems/ESS/Country/10_val_2_act/PVS/" -prip_dir "value_systems/ESS/Country/10_val_2_act/PriP/" -n_values 10 -n_actions 2 -e 1e-4 -output_dir "/results/ESS_COUNTRY/10_val_2_acts/"
## TODO: Insert data analysis

# Experiment ``ESS Individual-level'': Run on ESS data, with no abstraction (default)
## Data
#python -O runner.py -pvs_dir $ess_ind_F -prip_dir $ess_ind_PF -n_values 4 -n_actions 2 -e 1e-4 -output_dir "/results/ESS_INDIVIDUAL/"
## Data Analysis
#python data_analysis/data_analysis_main.py

###### SYNTHETIC DATA EXP. ######

# Experiment ``vary_grp_fact'''
## Data
#python -O runner.py -pvs_dir "value_systems/Synthetic/vary_grp_fact/PVS/" -prip_dir "value_systems/Synthetic/vary_grp_fact/PriP/" -n_values 4 -n_actions 3 -output_dir "/results/SYNTH_vary_grp_fact/"
## Data Analysis
python data_analysis/data_analysis_main.py -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/SYNTH_vary_grp_fact/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_grp_fact/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_grp_fact/PriP/" -output_dir "plots/SYNTH_vary_grp_fact/"


###### ENVIRONMENT VARIABLES, SYNTHETIC DATA EXP. ######




# Experiment 2: Run on VALE data -> Nice for loop!
## Data
#python main.py -f vale_f -pf vale_pf -slm -t -range -hcva -hcva2 -n_values 2 -n_actions 1
## Data with synth principle sets
#for item in $vale_synth_pfs; do
#  python main.py -f vale_f -pf item -slm -hcva -hcva2 -n_values 2 -n_actions 1
#done