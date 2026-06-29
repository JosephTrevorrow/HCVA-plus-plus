import pandas as pd
import copy
from plot_fairness import gini_coefficient, calc_envy_freeness, check_maximin_fairness, plot_residuals
from plot_utility import plot_pareto_efficiency, plot_total_utility
#from plot_limits import plot_data
#from data_analysis.plot_principles import *
import argparse as ap
from datetime import datetime as dt
import os

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-cons_dir', type=str, help='Directory pointing to the consensus files used in the experiment')

    parser.add_argument('-agents_pvs_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-agents_prip_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    ## ENV ARGS
    #parser.add_argument("-n_values", nargs="*", type=int, default=[4], help='Number of values')
    #parser.add_argument("-n_actions",nargs="*", type=int, default=3, help='Number of actions')
    #parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.
    args = parser.parse_args()

    now = dt.now().isoformat()

    # I want to run data analysis for 1 set of experiments. That means 1 or more agent files (PriP and PVS), and their
    # corresponding consensus for each run.

    ## Begin by finding a list of all consensus pvs and prips, for agents and cons.
    cons_pvs_sets = []
    cons_dir = args.cons_dir
    for file in os.listdir(cons_dir):
        if file.endswith(".csv") and "PERSONAL" in file:
            cons_pvs_sets.append(cons_dir + file)
    len_of_pvs_sets = len(cons_pvs_sets)
    pvs = cons_pvs_sets[0]
    print("cons_pvs_sets: ", cons_pvs_sets, "")
    cons_prip_sets = []
    cons_dir = args.cons_dir
    for file in os.listdir(cons_dir):
        if file.endswith(".csv") and "METADATA" in file:
            cons_prip_sets.append(cons_dir + file)
    len_of_prip_sets = len(cons_prip_sets)
    # Ensure this is sorted by i's
    #cons_prip_sets.sort() CHECK IF THIS IS THE CASE
    print("Cons_prip_sets: ", cons_prip_sets, "")
    prip = cons_prip_sets[0]

    # Len of agents_pvs will be same as agents prips.
    agents_pvs_sets = []
    agents_pvs_dir = args.agents_pvs_dir
    for file in os.listdir(agents_pvs_dir):
        if file.endswith(".csv"):
            agents_pvs_sets.append(agents_pvs_dir + file)
    len_of_agents_pvs_sets = len(agents_pvs_sets)
    print("agents_pvs_sets: ", agents_pvs_sets, "")
    agents = agents_pvs_sets[0]

    agents_prip_sets = []
    agents_prip_dir = args.agents_prip_dir
    for file in os.listdir(agents_prip_dir):
        if file.endswith(".csv"):
            agents_prip_sets.append(agents_prip_dir + file)
    len_of_agents_prip_sets = len(agents_prip_sets)
    print("agents_prip_sets: ", agents_prip_sets, "")
    agents = agents_prip_sets[0]

    ## Big for loop here running evaluation for each of the PVS sets
    for i in range(0, max(len(cons_pvs_sets), len(cons_prip_sets))):
        print("Iteration:", i, "of {}".format(max(len(cons_pvs_sets), len(cons_prip_sets))))
        ## update the pvs and prip, for cons and for agents
        if i < len(cons_pvs_sets):
            print("Cons PVS: ", i)
            pvs = cons_pvs_sets[i]
        if i < len(cons_prip_sets):
            print("Cons PriP: ", i)
            prip = cons_prip_sets[i]
        if i < len(agents_pvs_sets):
            print("Agents PVS: ", i)
            agents_pvs = agents_pvs_sets[i]
        if i < len(agents_prip_sets):
            print("Agents PriP: ", i)
            agents_prip = agents_prip_sets[i]

        list_of_params = ["P", "VA", "PriP"]

        # Load in the dataframes and combine
        cons_df = pd.read_csv(pvs)
        prips_df = pd.read_csv(prip)
        cons_df = pd.concat([cons_df, prips_df], axis=1, join="inner")

        agents_pvs_df = pd.read_csv(agents_pvs)
        agents_prip_df = pd.read_csv(agents_prip)
        agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")

        # Uses cons_df to unpack the values and actions lists, but cols for both dfs should be identical
        values_list, actions_list, principles_list = [], [], []
        if "P" in list_of_params:
            # Instead of doing this, you could load it into a matrix and cut down the diagonal, then unpack?
            # Drop P from list_of_params
            list_of_params.remove("P")
            values_list = list([col for col in cons_df.columns if 'P__' in col])
            # Clean list_of_params
            # Remove all cols that have the same two values (P__Universalism__Universalism, P__Benevolence__Benevolence, etc.)
            for col in values_list:
                col_split = col.split("__")
                if len(col_split) == 3 and col_split[1] == col_split[2]:
                    values_list.remove(col)
                else:
                    # Not dropped, so drop the symmetrical col (P__A__B == P__B__A)
                    symmetrical_col = "P__" + col_split[2] + "__" + col_split[1]
                    if col in values_list:
                        values_list.remove(col)
        if "VA" in list_of_params:
            list_of_params.remove("VA")
            actions_list = list([col for col in cons_df.columns if 'VA__' in col])
        if "PriP" in list_of_params:
            list_of_params.remove("PriP")
            principles_list = ['Egalitarian']

        # Add the unpacked lists to the list_of_params. Now we have 3 lists of the params left in our dfs
        list_of_params.extend(values_list)
        list_of_params.extend(actions_list)
        list_of_params_without_prip = copy.copy(list_of_params)
        list_of_params.extend(principles_list)

        # Filter cons and agents_df considering list_of_params
        # cons_df will have "Egalitarian" columns that are empty.
        cons_df_cols = copy.copy(list_of_params)
        cons_df_cols.append("p")
        cons_df = cons_df[cons_df_cols]

        agents_df_cols = copy.copy(list_of_params)
        agents_df_cols.extend(["country"])
        agents_df = agents_df[list_of_params]

        # Plot and run analysis

        ## RESIDUALS
        # Note: list_of_params = all parameters

        ### PVS Residuals
        plot_residuals(cons_df, agents_df, list_of_params_without_prip, "Entire PVS Residuals")
        # Just VAs
        plot_residuals(cons_df, agents_df, actions_list, "VAs Residuals")
        # Just Ps
        plot_residuals(cons_df, agents_df, values_list, "Ps Residuals")

        ### PriPs Residuals
        plot_residuals(cons_df, agents_df, principles_list, "PriPs Residuals")

        # PVSs and PriPs
        plot_residuals(cons_df, agents_df, list_of_params, "PVSs and PriPs Residuals")

        ## GINI
        ### PVS
        gini_coefficient(cons_df, agents_df, list_of_params)

        ## UTILITY
        #plot_pareto_efficiency(cons_df, agents_df, list_of_params)
        plot_total_utility(cons_df, agents_df, list_of_params)

        ## PRIP SENSITIVITY??


