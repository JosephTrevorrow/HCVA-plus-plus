
# Globals
PY_FILENAME="main.py"
F="value_systems/ESS_ABSTRACTED_value_system.csv"
PF="input_data/principles/placeholder_principles.csv"
SLMF="input_data/sml_principles/placeolder_sml.csv"


# Experiment 1: Find the transition point (Debug check) (Synthetic/CASE1_**.csv)
python PY_FILENAME -f $F -t -range -n_values 4 -n_actions 2

# Experiment 1: Do everything and store.
