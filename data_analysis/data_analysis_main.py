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
    parser.add_argument('-pvs_dir', type=str, help='Directory pointing to the prips csvs used in the experiment')
    parser.add_argument('-prip_dir', type=str, help='Directory pointing to the prips csvs used in the experiment')
    parser.add_argument('-agents_pvs_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-agents_prip_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    ## ENV ARGS
    #parser.add_argument("-n_values", nargs="*", type=int, default=[4], help='Number of values')
    #parser.add_argument("-n_actions",nargs="*", type=int, default=3, help='Number of actions')
    #parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.
    args = parser.parse_args()

    # Load in dataframe
    now = dt.now().isoformat()

    ## Begin by finding a list of all pvs and prips, for agents and cons
    pvs_sets = []
    pvs_dir = args.pvs_dir
    for file in os.listdir(pvs_dir):
        if file.endswith(".csv"):
            pvs_sets.append(pvs_dir + file)
    len_of_pvs_sets = len(pvs_sets)
    pvs = pvs_sets[0]

    prip_sets = []
    prip_dir = args.prip_dir
    for file in os.listdir(prip_dir):
        if file.endswith(".csv"):
            prip_sets.append(prip_dir + file)
    len_of_prip_sets = len(prip_sets)
    prip = prip_sets[0]

    # Len of agents_pvs will be same as agents prips, so just do one.
    agents_pvs_sets = []
    agents_pvs_dir = args.agents_pvs_dir
    for file in os.listdir(agents_pvs_dir):
        if file.endswith(".csv"):
            agents_pvs_sets.append(agents_pvs_dir + file)
    len_of_agents_pvs_sets = len(agents_pvs_sets)
    agents = agents_pvs_sets[0]

    agents_prip_sets = []
    agents_prip_dir = args.agents_prip_dir
    for file in os.listdir(agents_prip_dir):
        if file.endswith(".csv"):
            agents_prip_sets.append(agents_prip_dir + file)
    len_of_agents_prip_sets = len(agents_prip_sets)
    agents = agents_prip_sets[0]

    ## Big for loop here running evaluation for each of the PVS sets
    for i in range(0, max(len(pvs_sets), len(prip_sets))):
        print("Iteration: {}".format(max(len(pvs_sets), len(prip_sets))))
        ## update the pvs and prip, for cons and for agents
        if i < len(pvs_sets):
            print("PVS: ", i)
            pvs = pvs_sets[i]
        if i < len(prip_sets):
            print("PriP: ", i)
            prip = prip_sets[i]
        if i < len(agents_pvs_sets):
            print("Agents PVS: ", i)
            agents_pvs = agents_pvs_sets[i]
        if i < len(agents_prip_sets):
            print("Agents PriP: ", i)
            agents_prip = agents_prip_sets[i]

        list_of_params = ["P", "VA", "PriP"]

        # Load in the dataframes
        cons_df = pd.read_csv(pvs)
        prips_df = pd.read_csv(prip)
        agents_pvs_df = pd.read_csv(agents_pvs)
        agents_prip_df = pd.read_csv(agents_prip)

        # Uses cons_df to unpack the values and actions lists, but cols for both dfs should be identical
        values_list, actions_list, principles_list = [], [], []
        if "P" in list_of_params:
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
            principles_list = ['Egaliatarianism']

        # Add the unpacked lists to the list_of_params
        list_of_params.extend(values_list)
        list_of_params.extend(actions_list)
        list_of_params.extend(principles_list)

        # Filter cons and agents_df considering list_of_params
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
        plot_residuals(cons_df, agents_df, list_of_params, "Entire PVS Residuals")
        # Just VAs
        plot_residuals(cons_df, agents_df, actions_list, "VAs Residuals")
        # Just Ps
        plot_residuals(cons_df, agents_df, values_list, "Ps Residuals")

        ### PriPs Residuals
        # TODO, This isn't showing properly
        plot_residuals(cons_df, agents_df, principles_list, "PriPs Residuals")

        # PVSs and PriPs
        total_list = values_list + actions_list + principles_list
        plot_residuals(cons_df, agents_df, total_list, "PVSs and PriPs Residuals")

        ## GINI
        ### PVS
        gini_coefficient(cons_df, agents_df, list_of_params)

        ## UTILITY
        #plot_pareto_efficiency(cons_df, agents_df, list_of_params)
        plot_total_utility(cons_df, agents_df, list_of_params)

        ## PRIP SENSITIVITY


