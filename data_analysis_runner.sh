###### ESS DATA EXP. ######

# Experiment ``ESS Country-level'': Run on ESS data, at a country level abstraction (default)
#python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/4_val_2_act/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_2_act/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_2_act/PriP/" -output_dir "/ESS_4_val_2_act/"

#python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/4_val_3_act/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PriP/" -output_dir "/ESS_4_val_3_act/"

#python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/10_val_2_act/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/10_val_2_act/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/10_val_2_act/PriP/" -output_dir "/ESS_10_val_2_act/"

##### VALE EXPERIMENTS, VALIDATION ##########

python data_analysis/data_analysis_main.py -time_series_plots True -single_timestep_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/VALE/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/VALE/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/VALE/PriP/" -output_dir "/VALE/"

###### SYNTHETIC DATA EXP. ######

# Experiment ``vary_grp_fact'''
python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/SYNTH_vary_grp_fact/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_grp_fact/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_grp_fact/PriP/" -output_dir "/SYNTH_vary_grp_fact/"

# Experiment ``vary_mup_vamu'''
python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/SYNTH_vary_mup_vamu/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_mup_vamu/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_mup_vamu/PriP/" -output_dir "/SYNTH_vary_mup_vamu/"

# Experiment ``vary_pvs_prip'''
python data_analysis/data_analysis_main.py -time_series_plots True -cons_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/SYNTH_vary_pvs_prip/" -agents_pvs_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_pvs_prip/PVS/" -agents_prip_dir "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/vary_pvs_prip/PriP/" -output_dir "/SYNTH_vary_pvs_prip/"


