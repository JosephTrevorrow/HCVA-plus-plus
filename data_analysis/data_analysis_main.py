import re

import pandas as pd
import copy
from eval_plots import *
import argparse as ap
from datetime import date
import os
import fnmatch
import glob

## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

def plot_ess(now, args):
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

        ## Grab the agents PVS and concat them. ag_pvs and ag_prip will always be lists of size 1
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
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "Entire PVS Residuals", args)
        ### Just VAs
        plot_residuals(normalised_cons_sets, normalised_agents_df, actions_list, "VAs Residuals", args)
        ### Just Ps
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list, "Ps Residuals", args)

        ### PriPs Residuals
        plot_residuals(normalised_cons_sets, normalised_agents_df, ['Egalitarian'], "PriPs Residuals", args)

        # PVSs and PriPs
        plot_residuals(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list+['Egalitarian'], "PVSs and PriPs Residuals", args)

        ## GINI
        ### PVS
        gini_coefficient(normalised_cons_sets, normalised_agents_df, cleaned_values_list+actions_list, "PVSs_and_PriPs.csv", args)
    return

def plot_synth(args, experiment_name):
    """This method computes plots from synthetic data"""

    # Find output_dirs/sum
    list_of_output_dirs = []
    for fname in os.listdir(args.cons_dir):
        # HCVA here as a filter for a single method. All methods will have same len.
        if re.search(r"HCVA_PERSONALS_DIR_[0-9]+_RUN_"+str(args.steps)+"_", fname):
            # get the dir num from the fname
            list_of_output_dirs.append(fname.split("_")[3])
    # CAUTION: Cutting for debug
    #list_of_output_dirs = list_of_output_dirs[:2]

    # Construct dir_dict of form {dir_id: [normalised_cons, normalised_agents], ...}
    dir_dict = {}
    for dir in list_of_output_dirs:
        agents_list = []
        # Given a directory where a consensus is, find all consensuses in dir.
        cons_pvs_sets = []
        cons_prip_sets = []
        for fname in os.listdir(args.cons_dir):
            if re.search(r"_PERSONALS_DIR_" + str(dir) +"_RUN_[0-9]+", fname):
                cons_pvs_sets.append(args.cons_dir+fname)
            elif re.search(r"_METADATA_DIR_" + str(dir) +"_RUN_[0-9]+", fname):
                cons_prip_sets.append(args.cons_dir+fname)
        # Sort: (Note, these are filenames for cons sets)
        cons_pvs_sets = sort_nicely(cons_pvs_sets)
        cons_prip_sets = sort_nicely(cons_prip_sets)

        # For each cons found, load into a df, store in a dict of dfs with baseline names.
        # For loop returns a dict cons_sets, which has <len(baselines)> dfs, 1 for each baseline. The df will contain <len(timesteps)> rows, 1 per cons.
        cons_sets = {}
        for j in range(0, len(cons_pvs_sets)):
            cons_pvs = pd.read_csv(cons_pvs_sets[j])
            cons_prip = pd.read_csv(cons_prip_sets[j])
            # Get the key (=baseline name) (the text before the first _ in the filename)
            key = os.path.splitext(os.path.basename(cons_pvs_sets[j]))[0].split("_")[0]
            # Removed PriP cols other than preference
            cons_prip = cons_prip[['Egalitarian']]
            cons_df = pd.concat([cons_pvs, cons_prip], axis=1, join="inner")
            # Add cons_df to cons_sets dict
            if key not in cons_sets:
                cons_sets[key] = copy.deepcopy(cons_df)
            else:
                cons_sets[key] = pd.concat([cons_sets[key], copy.deepcopy(cons_df)])

        # input every single agent, place in a list of df (form: [t1_agents, t2_agents, etc.
        ag_pvs = glob.glob(args.agents_pvs_dir + str(dir)+"/"+"*PVS*")
        ag_prip = glob.glob(args.agents_prip_dir + str(dir)+"/"+"*PriP*")
        sorted_ag_pvs = sort_nicely(ag_pvs)
        sorted_ag_prip = sort_nicely(ag_prip)

        for i in range(0, len(sorted_ag_pvs)):
            agents_pvs_df = pd.read_csv(sorted_ag_pvs[i])
            agents_prip_df = pd.read_csv(sorted_ag_prip[i])
            agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")
            agents_list.append(copy.deepcopy(agents_df))

        ## remove the irrelevant cols from every single df you've just sorted out. Create a list of params to use with residuals
        #  Every df will have the same cols, so we find them for one, and copy this
        # note, we will use values_list and actions_list to filter our data analysis plots.
        values_list = list([col for col in cons_sets["HCVA"].columns if 'P__' in col])
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
        ## Again, because every method will have the exact same col names, we just use HCVA here.
        actions_list = list([col for col in cons_sets["HCVA"].columns if 'VA__' in col])
        agents_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]
        cons_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]

        # Filter using cols_to_keeps (Agents list and cons list will be the same)
        for i in range(0, len(agents_list)):
            agents_list[i] = agents_list[i][agents_cols_to_keep]
        cons_sets = {k: v[cons_cols_to_keep] for k, v in cons_sets.items()}

        normalised_cons_sets = normalise_cons_time_series(cons_sets)
        normalised_agents_df = normalise_agents_time_series(agents_list)
        dir_dict[int(dir)] = [normalised_cons_sets, normalised_agents_df]



    print("Time to plot!")
    ## STEP 2: use dir_dict to find the mean of resiudals
    # PVS
    plot_mean_residuals(dir_dict, cleaned_values_list + actions_list, experiment_name+"pvs_100_runs_residual", output_dir=args.output_dir)
    ### Just VAs
    plot_mean_residuals(dir_dict, actions_list, experiment_name+"va_100_runs_residual", output_dir=args.output_dir)
    ### Just Ps
    plot_mean_residuals(dir_dict, cleaned_values_list, experiment_name+"p_100_runs_residual", output_dir=args.output_dir)
    ### PriPs Residuals
    plot_mean_residuals(dir_dict, ['Egalitarian'], experiment_name+"prip_100_runs_residual", output_dir=args.output_dir)
    # PVSs and PriPs
    plot_mean_residuals(dir_dict, cleaned_values_list + actions_list + ['Egalitarian'], experiment_name+"pvs_prip_100_runs_residual", output_dir=args.output_dir)

    ## GINI
    ### PVS
    plot_mean_gini(dir_dict, cleaned_values_list + actions_list, title=experiment_name+"pvs_gini",output_dir=args.output_dir)
    plot_mean_gini(dir_dict, actions_list, title=experiment_name+"actions_gini",output_dir=args.output_dir)
    plot_mean_gini(dir_dict, cleaned_values_list, title=experiment_name+"prefs_gini",output_dir=args.output_dir)
    plot_mean_gini(dir_dict, ['Egalitarian'], title=experiment_name+"prips_gini",output_dir=args.output_dir)
    plot_mean_gini(dir_dict, cleaned_values_list + actions_list + ["Egalitarian"], title=experiment_name+"pvs_prips_gini",output_dir=args.output_dir)

    ## Individual agents
    plot_violin_individuals(dir_dict, cleaned_values_list+actions_list, title=experiment_name+"pvs_residuals_individuals", output_dir=args.output_dir, difference=True)
    plot_violin_individuals(dir_dict, actions_list, title=experiment_name+"violin", output_dir=args.output_dir+"actions_residuals_individuals", difference=True)
    plot_violin_individuals(dir_dict, cleaned_values_list, title=experiment_name+"prefs_residuals_individuals", output_dir=args.output_dir, difference=True)
    plot_violin_individuals(dir_dict, ["Egalitarian"], title=experiment_name+"prips_residuals_individuals", output_dir=args.output_dir, difference=True)
    plot_violin_individuals(dir_dict, cleaned_values_list+actions_list+["Egalitarian"], title=experiment_name+"pvs_prips_residuals_individuals", output_dir=args.output_dir, difference=True)
    return

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-ess', action='store_true', help='Whether to run the single timestep plots')
    parser.add_argument('-synth', action='store_true', help='Whether to run the time series plots')

    parser.add_argument('-steps', type=str, default="0", help='Number of steps in an experiment')

    parser.add_argument('-cons_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/4_val_3_act/", help='Directory pointing to the consensus files used in the experiment')
    parser.add_argument('-agents_pvs_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PVS/", help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-agents_prip_dir', type=str,default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/value_systems/ESS/Country/4_val_3_act/PriP/", help='Directory pointing to the agents csvs used in the experiment')
    parser.add_argument('-output_dir', type=str, default="/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/", help='Directory to save the output files')
    args = parser.parse_args()
    now = str(date.today())
    experiment_name = os.path.basename(str(args.cons_dir))
    if args.ess:
        plot_ess(now, args)
    elif args.synth:
        plot_synth(args, experiment_name)