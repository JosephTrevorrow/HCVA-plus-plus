
# Globals
ess_F="value_systems/ESS/Country/PVS/"
ess_PF="value_systems/ESS/Country/PriP/"

ess_ind_F="value_systems/ESS/Individual/PVS/"
ess_ind_PF="value_systems/ESS/Individual/PriP/"


synthetic_1_f=""
synthetic_1_pf=""

vale_f="value_systems/VALE/PVS.csv"
vale_pf="value_systems/VALE/PriP.csv"
vale_synth_pfs='a.txt b.txt c.txt d.txt'

###### ESS DATA EXP. ######
echo $ess_F
# Experiment ``ESS Country-level'': Run on ESS data, at a country level abstraction (default)
## Data
python runner.py -pvs_dir $ess_F -prip_dir $ess_PF -n_values 4 -n_actions 2 -e 1e-4 -output_dir "results/ESS_COUNTRY/"
## Data Analysis
#python data_analysis/data_analysis_main.py

# Experiment ``ESS Individual-level'': Run on ESS data, with no abstraction (default)
## Data
python main.py -f $ess_ind_F -pf $ess_ind_PF -t -range -hcva -hcva2 -n_values 4 -n_actions 2 -output_dir "results/ESS_INDIVIDUAL/"
## Data Analysis
#python data_analysis/data_analysis_main.py


###### SYNTHETIC DATA EXP. ######

## Note: Synthetic data experiments are ran over a range of different paramters, depending on the experiment. Therefore, they use the runner.py file that handles this (incl. storage)

# Experiment ``Maj/min split'': Run on ESS data, at a country level abstraction (default)
## Data
#python main.py -f ess_F -pf ess_PF -t -range -hcva -hcva2 -n_values 4 -n_actions 2
## Data Analysis
#python data_analysis/data_analysis_main.py


###### ENVIRONMENT VARIABLES, SYNTHETIC DATA EXP. ######




# Experiment 2: Run on VALE data -> Nice for loop!
## Data
#python main.py -f vale_f -pf vale_pf -slm -t -range -hcva -hcva2 -n_values 2 -n_actions 1
## Data with synth principle sets
#for item in $vale_synth_pfs; do
#  python main.py -f vale_f -pf item -slm -hcva -hcva2 -n_values 2 -n_actions 1
#done