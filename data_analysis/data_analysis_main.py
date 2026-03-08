"""This file runs all methods of data analysis, found in the data_analysis folder"""

# TODO: Fix residual boxplots
# TODO: Fix Envy Freeness

import pandas as pd
import copy
from plot_fairness import gini_coefficient, calc_envy_freeness, check_maximin_fairness, plot_residuals
from plot_utility import plot_pareto_efficiency, plot_total_utility
#from plot_limits import plot_data
#from data_analysis.plot_principles import *

if __name__ == "__main__":
    # Load in dataframe
    cons_df = pd.read_csv("/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/synthetic_results/CASE2xPriP1/CASE2_PriP1_BASELINES_PVS.csv")
    agents_df = pd.read_csv("/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/Synthetic/CASE2_PVS.csv")

    list_of_params = ["P", "VA"]
    title = "Pretty cold for a placehold"

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

    ## FAIRNESS

    ### Residuals (list_of_params is all params
    plot_residuals(cons_df, agents_df, list_of_params, "Entire PVS Residuals")
    # Just VAs
    plot_residuals(cons_df, agents_df, actions_list, "VAs Residuals")
    # Just Ps
    plot_residuals(cons_df, agents_df, values_list, "Ps Residuals")
    # Just PriPs
    # TODO, This isn't showing properly
    plot_residuals(cons_df, agents_df, principles_list, "PriPs Residuals")
    # PVSs and PriPs
    total_list = values_list + actions_list + principles_list
    plot_residuals(cons_df, agents_df, total_list, "PVSs and PriPs Residuals")

    check_maximin_fairness(cons_df, agents_df, list_of_params)
    gini_coefficient(cons_df, agents_df, list_of_params)
    calc_envy_freeness(cons_df, agents_df, list_of_params)

    ## UTILITY
    plot_pareto_efficiency(cons_df, agents_df, list_of_params)
    plot_total_utility(cons_df, agents_df, list_of_params)

    ## PRINCIPLES


    ## STABILITY
