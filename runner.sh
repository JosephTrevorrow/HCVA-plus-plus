
# Globals
ess_F="value_systems/ESS/PVS_abstracted.csv"
ess_PF="value_systems/ESS/3q_PriP.csv"

synthetic_1_f=""
synthetic_1_pf=""

vale_f="value_systems/VALE/PVS.csv"
vale_pf="value_systems/VALE/PriP.csv"
vale_synth_pfs='a.txt b.txt c.txt d.txt'


# Experiment 1: Run on ESS data
## Data
python main.py -f ess_F -pf ess_PF -t -range -hcva -hcva2 -n_values 4 -n_actions 2
## Data Analysis
python data_analysis/data_analysis_main.py

# Experiment 2: Run on VALE data
## Data
python main.py -f vale_f -pf vale_pf -slm -t -range -hcva -hcva2 -n_values 2 -n_actions 1
## Data with synth principle sets
for item in $vale_synth_pfs; do
  python main.py -f vale_f -pf item -slm -hcva -hcva2 -n_values 2 -n_actions 1
done

## Data Analysis
python data_analysis/data_analysis_main.py

# Experiment 3: Run on synthetic data
