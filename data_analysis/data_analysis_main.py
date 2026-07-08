import pandas as pd
import copy
from plot_fairness import gini_coefficient, calc_envy_freeness, check_maximin_fairness, plot_residuals
from plot_utility import plot_pareto_efficiency, plot_total_utility
#from plot_limits import plot_data
#from data_analysis.plot_principles import *
import argparse as ap
from datetime import date
import os
import fnmatch
import glob

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-cons_dir', type=str, help='Directory pointing to the consensus files used in the experiment')
    parser.add_argument('-agents_pvs_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-agents_prip_dir', type=str, help='Directory pointing to the agents csvs used in the experiment')
    args = parser.parse_args()

    now = str(date.today())
    # Find the number of pvs files (this, or any other list of files will tell us the number of iterations)
    count = len(fnmatch.filter(os.listdir(args.agents_pvs_dir), "*.csv"))

    ## Big for loop here running evaluation for each of the PVS sets
    for i in range(0, count):
        print("Iteration:", i, "of ", count)
        ## update the pvs and prip, for cons and for agents
        # these will be nested lists where each item in the list is a filename for a cons_dir for this iteration.
        cons_sets = {}
        ## Grab all Cons PVS and all Cons PriP where "__"+i+"__ is in filename.
        cons_pvs_sets = glob.glob(args.cons_dir + "*_PERSONALS_"+str(i)+"*")
        cons_prip_sets = glob.glob(args.cons_dir + "*_METADATA_"+str(i)+"*")
        print("cons_pvs_sets: ", cons_pvs_sets, "cons_prip_sets: ", cons_prip_sets, "")
        # for each of these, load them into a df, and store in a dict of dfs with baseline names
        for j in range(0, len(cons_pvs_sets)):
            cons_pvs = pd.read_csv(cons_pvs_sets[j])
            cons_prip = pd.read_csv(cons_prip_sets[j])
            # Get the key (the text before the first _ in the filename)
            key = os.path.splitext(os.path.basename(cons_pvs_sets[j]))[0].split("_")[0]
            # Removed prip cols other than preference
            cons_prip = cons_prip[['Egalitarian']]
            cons_df = pd.concat([cons_pvs, cons_prip], axis=1, join="inner")
            ## Add cons_df to cons_sets dict
            cons_sets[key] = cons_df

        ## Grab the agents PVS and concat them
        ag_pvs = glob.glob(args.agents_pvs_dir + "*_PVS_"+str(i)+".csv")
        ag_prip = glob.glob(args.agents_prip_dir + "*_PriP_"+str(i)+".csv")
        print(args.agents_pvs_dir + "*_PVS_"+str(i)+".csv")
        print("ag_pvs: ", ag_pvs, "ag_prip: ", ag_prip, "")
        agents_pvs_df = pd.read_csv(ag_pvs[0])
        agents_prip_df = pd.read_csv(ag_prip[0])
        agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")

        # Remove irrelevant cols from every df. Every df will have the same cols, so we find them for one, and copy this
        # note, we will use values_list and actions_list to filter our data analysis plots.
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
        actions_list = list([col for col in cons_df.columns if 'VA__' in col])
        agents_cols_to_keep = values_list + actions_list + ["country"] + ["Egalitarian"]
        cons_cols_to_keep = values_list + actions_list + ["Egalitarian"] + ["p"]

        # cons_sets will have "Egalitarian" columns that are empty. Will need to account for this.
        cons_sets = {k: v[cons_cols_to_keep] for k, v in cons_sets.items()}
        agents_df = agents_df[agents_cols_to_keep]

        # Plot and run analysis

        ## RESIDUALS

        ### PVS Residuals
        plot_residuals(cons_sets, agents_df, values_list+actions_list, "Entire PVS Residuals")
        ### Just VAs
        plot_residuals(cons_sets, agents_df, actions_list, "VAs Residuals")
        ### Just Ps
        plot_residuals(cons_sets, agents_df, values_list, "Ps Residuals")

        ### PriPs Residuals
        plot_residuals(cons_sets, agents_df, ['Egalitarian'], "PriPs Residuals")

        # PVSs and PriPs
        plot_residuals(cons_sets, agents_df, values_list+actions_list+['Egalitarian'], "PVSs and PriPs Residuals")

        ## GINI
        ### PVS
        gini_coefficient(cons_sets, agents_df, values_list+actions_list, "PVSs_and_PriPs.csv")

        ## TODO: Add more here.

        ## UTILITY
        #plot_pareto_efficiency(cons_df, agents_df, list_of_params)
        plot_total_utility(cons_sets, agents_df, values_list+actions_list, "Total Utility")

        ## PRIP SENSITIVITY??


