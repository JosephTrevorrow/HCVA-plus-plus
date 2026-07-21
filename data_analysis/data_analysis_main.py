import pandas as pd
import copy
from plot_fairness import *
from plot_utility import plot_pareto_efficiency, plot_total_utility, plot_utility_over_time
#from plot_limits import plot_data
#from data_analysis.plot_principles import *
import argparse as ap
from datetime import date
import os
import fnmatch
import glob

def single_timestep_graphs(now, args):
    # Find the number of pvs files (this, or any other list of files will tell us the number of iterations)
    count = len(fnmatch.filter(os.listdir(args.agents_pvs_dir), "*.csv"))

    ## Big for loop here running evaluation for each of the PVS sets with each of the results sets
    for i in range(0, count):
        print("Iteration:", i, "of ", count)
        ## update the pvs and PriP, for cons and for agents
        # these will be nested lists where each item in the list is a filename for a cons_dir for this iteration.
        cons_sets = {}
        ## Grab all Cons PVS and all Cons PriP where "__"+i+"__ is in filename.

        # Grab the cons sets for this count
        cons_pvs_sets = glob.glob(args.cons_dir + "*_PERSONALS_"+str(i)+"*")
        cons_prip_sets = glob.glob(args.cons_dir + "*_METADATA_"+str(i)+"*")
        cons_pvs_sets = sorted(cons_pvs_sets)
        cons_prip_sets = sorted(cons_prip_sets)

        # for each of the cons, load them into a df, and store in a dict of dfs with baseline names
        for j in range(0, len(cons_pvs_sets)):
            cons_pvs = pd.read_csv(cons_pvs_sets[j])
            cons_prip = pd.read_csv(cons_prip_sets[j])
            # Get the key (the text before the first _ in the filename)
            key = os.path.splitext(os.path.basename(cons_pvs_sets[j]))[0].split("_")[0]
            # Removed PriP cols other than preference
            cons_prip = cons_prip[['Egalitarian']]
            cons_df = pd.concat([cons_pvs, cons_prip], axis=1, join="inner")
            ## Add cons_df to cons_sets dict
            cons_sets[key] = copy.deepcopy(cons_df)

        ## Grab the agents PVS and concat them
        ag_pvs = glob.glob(args.agents_pvs_dir + "*PVS*")
        ag_prip = glob.glob(args.agents_prip_dir + "*PriP*")
        agents_pvs_df = pd.read_csv(ag_pvs[0])
        agents_prip_df = pd.read_csv(ag_prip[0])
        agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")

        # Remove irrelevant cols from every df. Every df will have the same cols, so we find them for one, and copy this
        # note, we will use values_list and actions_list to filter our data analysis plots.
        values_list = list([col for col in cons_df.columns if 'P__' in col])
        cleaned_values_list = copy.deepcopy(values_list)
        # Clean list_of_params
        # Remove all cols that have the same two values (P__Universalism__Universalism, P__Benevolence__Benevolence, etc.)
        for col in values_list:
            col_split = col.split("__")
            if len(col_split) == 3 and col_split[1] == col_split[2]:
                cleaned_values_list.remove(col)
            elif col in cleaned_values_list:
                # Not dropped, so drop the symmetrical col (P__A__B == P__B__A)
                symmetrical_col = "P__" + col_split[2] + "__" + col_split[1]
                if symmetrical_col in cleaned_values_list:
                    cleaned_values_list.remove(symmetrical_col)
        print("cleaned values list: ", cleaned_values_list, "")
        actions_list = list([col for col in cons_df.columns if 'VA__' in col])
        #agents_cols_to_keep = cleaned_values_list + actions_list + ["country"] + ["Egalitarian"]
        agents_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]
        cons_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]

        # cons_sets will have "Egalitarian" columns that are empty. Will need to account for this.
        cons_sets = {k: v[cons_cols_to_keep] for k, v in cons_sets.items()}
        agents_df = agents_df[agents_cols_to_keep]

        ## Now we have a dict of the df for each method, and the agents df those methods aggregated.

        # Plot and run analysis

        ## RESIDUALS
        ## For residuals, normalise all values passed to between 0-1
        # Note that cons_sets is a dict of dfs, so we need to normalise each df separately
        normalised_cons_sets = normalise_cons(cons_sets)
        normalised_agents_df = normalise_agents(agents_df)

        ### PVS Residuals
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "Entire PVS Residuals", args.output_dir)
        ### Just VAs
        plot_residuals(normalised_cons_sets, normalised_agents_df, actions_list, "VAs Residuals", args.output_dir)
        ### Just Ps
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list, "Ps Residuals", args.output_dir)

        ### PriPs Residuals
        plot_residuals(normalised_cons_sets, normalised_agents_df, ['Egalitarian'], "PriPs Residuals", args.output_dir)

        # PVSs and PriPs
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list+['Egalitarian'], "PVSs and PriPs Residuals", args.output_dir)

        ## GINI
        ### PVS
        gini_coefficient(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "PVSs_and_PriPs.csv", args.output_dir)

        ## UTILITY - FIX THIS!
        #plot_pareto_efficiency(cons_df, agents_df, list_of_params)
        plot_total_utility(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "Total Utility", args.output_dir)
    return

def time_series_graphs(now, args):

    ## input every single consensus, place in a nested list[dict]? of df (form: [ [t1_hcva, t1_inf, t1_slm, etc.], [t2_hcva, t2_inf, t2_slm, etc.], etc.

    # using agents_pvs_dir here doesn't matter. Pvs, Prip and cons will all have the same len.
    count = len(fnmatch.filter(os.listdir(args.agents_pvs_dir), "*.csv"))
    consensus_list = []
    for i in range(0, count):
        cons_sets = {}
        # Grab the cons sets for this count
        cons_pvs_sets = glob.glob(args.cons_dir + "*_PERSONALS_" + str(i) + "*")
        cons_prip_sets = glob.glob(args.cons_dir + "*_METADATA_" + str(i) + "*")
        # for each of the cons, load them into a df, and store in a dict of dfs with baseline names
        for j in range(0, len(cons_pvs_sets)):
            cons_pvs = pd.read_csv(cons_pvs_sets[j])
            cons_prip = pd.read_csv(cons_prip_sets[j])
            # Get the key (the text before the first _ in the filename)
            key = os.path.splitext(os.path.basename(cons_pvs_sets[j]))[0].split("_")[0]
            # Removed PriP cols other than preference
            cons_prip = cons_prip[['Egalitarian']]
            cons_df = pd.concat([cons_pvs, cons_prip], axis=1, join="inner")
            ## Add cons_df to cons_sets dict
            cons_sets[key] = copy.deepcopy(cons_df)
        consensus_list.append(copy.deepcopy(cons_sets))

    ## input every single agent, place in a list of df (form: [t1_agents, t2_agents, etc.
    agents_list = []
    print("args.agents_pvs_dir: ", args.agents_pvs_dir)
    print("args.agents_prip_dir: ", args.agents_prip_dir)
    ## Grab the agents PVS and concat them
    ag_pvs = glob.glob(args.agents_pvs_dir + "*PVS*")
    ag_prip = glob.glob(args.agents_prip_dir + "*PriP*")
    sorted_ag_pvs = sorted(ag_pvs)
    sorted_ag_prip = sorted(ag_prip)
    print("sorted_ag_pvs: ", sorted_ag_pvs)
    print("sorted_ag_prip: ", sorted_ag_prip)

    for i in range(0, count):
        agents_pvs_df = pd.read_csv(sorted_ag_pvs[i])
        agents_prip_df = pd.read_csv(sorted_ag_prip[i])
        agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")
        agents_list.append(copy.deepcopy(agents_df))

    ## remove the irrelevant cols from every single df you've just sorted out. Create a list of params to use with residuals
    #  Every df will have the same cols, so we find them for one, and copy this
    # note, we will use values_list and actions_list to filter our data analysis plots.
    values_list = list([col for col in consensus_list[0]["HCVA"].columns if 'P__' in col])
    cleaned_values_list = copy.deepcopy(values_list)
    # Clean list_of_params
    # Remove all cols that have the same two values (P__Universalism__Universalism, P__Benevolence__Benevolence, etc.)
    for col in values_list:
        col_split = col.split("__")
        if len(col_split) == 3 and col_split[1] == col_split[2]:
            cleaned_values_list.remove(col)
        elif col in cleaned_values_list:
            # Not dropped, so drop the symmetrical col (P__A__B == P__B__A)
            symmetrical_col = "P__" + col_split[2] + "__" + col_split[1]
            if symmetrical_col in cleaned_values_list:
                cleaned_values_list.remove(symmetrical_col)
    print("cleaned values list: ", cleaned_values_list, "")
    ## Again, because every method will have the exact same col names, we just use HCVA here.
    actions_list = list([col for col in consensus_list[0]["HCVA"].columns if 'VA__' in col])
    agents_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]
    cons_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]

    # Filter using cols_to_keeps (Agents list and cons list will be the same)
    for i in range(0, len(agents_list)):
        agents_list[i] = agents_list[i][agents_cols_to_keep]
        consensus_list[i] = {k: v[cons_cols_to_keep] for k, v in consensus_list[i].items()}

    normalised_cons_sets = normalise_cons_time_series(consensus_list)
    normalised_agents_df = normalise_agents_time_series(agents_list)
    ## RESIDUALS
    ### PVS Residuals
    plot_residuals_over_time(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "Time Series PVS Residuals", dir=args.output_dir)
    ### Just VAs
    plot_residuals_over_time(normalised_cons_sets, normalised_agents_df, actions_list, "Time Series VAs Residuals", dir=args.output_dir)
    ### Just Ps
    plot_residuals_over_time(normalised_cons_sets, normalised_agents_df, cleaned_values_list, "Time Series Ps Residuals",dir=args.output_dir)
    ### PriPs Residuals
    plot_residuals_over_time(normalised_cons_sets, normalised_agents_df, ['Egalitarian'], "Time Series PriPs Residuals",dir=args.output_dir)
    # PVSs and PriPs
    plot_residuals_over_time(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list+['Egalitarian'], "Time Series PVSs and PriPs Residuals",dir=args.output_dir)
    ## GINI
    ### PVS
    plot_gini_over_time(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "Time Series Gini PVS",dir=args.output_dir)

    ## UTILITY
    # plot_pareto_efficiency(cons_df, agents_df, list_of_params)
    #plot_total_utility(cons_sets, agents_df, cleaned_values_list + actions_list, "Total Utility")
    # TODO: plot the utility over time
    return

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-single_timestep_plots', action='store_true', help='Whether to run the single timestep plots')
    parser.add_argument('-time_series_plots', action='store_true', help='Whether to run the time series plots')
    parser.add_argument('-cons_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/4_val_3_act/", help='Directory pointing to the consensus files used in the experiment')
    parser.add_argument('-agents_pvs_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PVS/", help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-agents_prip_dir', type=str,default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PriP/", help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-output_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/", help='Directory to save the output files')
    args = parser.parse_args()

    now = str(date.today())

    if args.single_timestep_plots:
        print("single timestep plots:")
        # you want single timestep plots for every method, for every iteration.
        single_timestep_graphs(now, args)
    elif args.time_series_plots:
        print("time series plots:")
        # you want time series plots for the synthetic methods where agent data has more than one iteration
        time_series_graphs(now, args)

    print("Done!")