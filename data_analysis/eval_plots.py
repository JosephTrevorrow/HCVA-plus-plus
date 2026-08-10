"""
This file will plot residuals given a consensus and a set of agents PVSs and PriPs. The gini index is also plotted
What makes up the residual can be set as an argument.
"""
import csv
import os
import numpy as np
import copy

def normalise_cons(sets):
    """Normalises the consensus data to be between 0-1. sets is of form: dict{key: df}"""
    normalised = {}
    for key, df in sets.items():
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised[key] = copy.deepcopy(df)
    return normalised

def normalise_agents(df):
    """Normalises the agent data to be between 0-1. sets is of form: df"""
    for column in df.columns:
        if 'VA__' in column:
            min_val = -1
            max_val = 1
        else:
            min_val = 0
            max_val = 1
        if column != 'p':
            df[column]= (df[column] - min_val) / (max_val - min_val)
    return df

def normalise_cons_time_series(sets):
    normalised = {}
    for key, df in sets.items():
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised[key] = copy.deepcopy(df)
    return normalised

def normalise_agents_time_series(sets):
    normalised = []
    for df in sets:
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        #df = df.astype(float)
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised.append(copy.deepcopy(df))
    return normalised

def gini_coefficient(cons_sets, agents_df, list_of_params, filename, args):
    """Calculates the Gini coefficient (Inequality of disappointment amongst agents) for a SINGLE TIMESTEP for ESS Data.
    Low total utility with High Gini means cons favours majority at expense of minority (low because lower is better)"""
    ginis = {}
    for key, cons_df in cons_sets.items():
        temp_residuals = np.array([], dtype=float)
        for agent in agents_df.iterrows():
            # For every col, match these two dfs
            temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df.iloc[0][list_of_params].to_numpy()).sum()
            temp_residuals = np.append(temp_residuals, [temp_residual])
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(temp_residuals, temp_residuals)).mean()
        # Relative mean absolute difference
        if np.mean(temp_residuals) == 0:
            print("WARNING: Mean of residuals is 0.")
        rmad = mad / np.mean(temp_residuals)
        # Gini coefficient
        g = 0.5 * rmad
        ginis[key] = copy.copy(g)
    ## Add to/Make a gini csv file and save ginis
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    with open(output_dir+filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(ginis.keys())
        writer.writerow(ginis.values())
    return

def plot_residuals(cons_sets, agents_df, list_of_params, title, args):
    """Plots a residual bar chart given a list of parameters using the dataframe. Style will follow prev. work.
    This function makes a boxplot chart, where each plot is a method. This is for one experiment, for ESS Data.
    X Axis: Ps, Y Axis: Residuals
    The plot includes bars for every baseline method"""
    boxplots = {}
    metadata = []
    # For every consensus, go through each agent, and find difference.
    for key, cons_df in cons_sets.items():
        points = []
        for agent in agents_df.iterrows():
            # For every col, match these two dfs and find the sum of aboslute difference between all of the vals to compare.
            temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df.iloc[0][list_of_params].to_numpy()).sum()
            points.append(copy.copy(temp_residual))
        boxplots[key] = copy.copy(points)
        # Get the last element of boxplots (the one we just made), find its var, std, mean, etc. and save to dict with p as title
        metadata.append({'key': key,'mean': np.mean(points), 'std': np.std(points), 'var': np.var(points), 'min': np.min(points), 'max': np.max(points), 'points':copy.copy(points)})
    """
    ## Make the boxplots
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(boxplots.values(), labels=boxplots.keys(), orientation='horizontal')
    """
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    #fig.savefig(output_dir+title+".png", bbox_inches="tight")
    # Now make sure you save the boxplot data (mean/IQR/Whiskers) in a csv, with info for that run
    with open(output_dir+title+"residuals.csv", 'w') as f:
        header = ['key', 'mean', 'std', 'var', 'min', 'max', 'points']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(metadata)

def find_and_store_residuals(dir_dict, list_of_params, title, output_dir, difference):
    """Finds the residual of an individual agent for every single timestep. Saves as a file
    - Y: Timestep (how each parameter is increasing (pvs_prip, mupvamu, grp_fact)
    - X: Residual or gini for a particular agent at that timestep, minus the gini or residual at the previous timestep.
    INPUTS:
    - dir_dict a dict of form {x : [normalised_cons_sets, normalised_agents_df]}, x=[0,100], x+=1
        - normalised_cons_sets is a list of dicts, [{hcva: cons, inf:cons}, {hcva: cons, inf: cons}, etc. ]
            - where each dict is all cons for a single timestep.
    - list_of_params: list of parameters to include in the residual calculation. e.g. [pvs, pvs+prip, etc. (listed as col names)]
    - output_dir: directory to save the plot to.
    - difference: bool that if True will make points = the difference in residual t and t-1, if False, then residuals are just the residuals for that timestep.
    """
    # Step 1: Create a dict `lines`, where we will store mean residuals
    lines = {}
    for key in dir_dict[list(dir_dict.keys())[0]][0].keys():
        # create subdicts for every agent, using a default initial agents set
        _, agents = dir_dict[list(dir_dict.keys())[0]]
        lines[key] = {}
        for i in range (0, len(agents[0])*len(dir_dict)): # Len of 1 runs agents * number of runs
            lines[key][i] = []

    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for run, (iteration, data) in enumerate(dir_dict.items()):
        # Unpack
        normalised_cons_sets, normalised_agents = data
        for key, cons_df in normalised_cons_sets.items():
            for timestep_id, agent in enumerate(normalised_agents):
                temp_residual = np.abs(
                    agent[list_of_params].to_numpy() - cons_df.iloc[timestep_id][list_of_params].to_numpy()).sum(axis=1)
                if difference and timestep_id > 0:
                    previous_residual = np.abs(
                        normalised_agents[timestep_id - 1][list_of_params].to_numpy() - cons_df.iloc[timestep_id - 1][
                            list_of_params].to_numpy()).sum(axis=1)
                for i in range(0, len(temp_residual)):
                    if difference and timestep_id > 0:
                        lines[key][i + (run * len(agents[0]))].append(temp_residual[i] - previous_residual[i])
                    else:
                        lines[key][i + (run * len(agents[0]))].append(temp_residual[i])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    ## Save all of lines to CSV
    for key, agents in lines.items():
        with open(output_dir + title + key +"violin.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "points"])
            for id, list_of_points in agents.items():
                row = [id] + list_of_points
                writer.writerow(row)
    return

def plot_mean_gini(dir_dict, list_of_params, title, output_dir):
    """Plots a line graph. Each line is the mean residual of a method over time. By time, we mean the varying of a parameter (e.g. increasing mu_p and mu_va).
            INPUTS:
            - dir_dict a dict of form {x : [normalised_cons_sets, normalised_agents_df]}, x=[0,100], x+=1
                - normalised_cons_sets is a list of dicts, [{hcva: cons, inf:cons}, {hcva: cons, inf: cons}, etc. ]
                    - where each dict is all cons for a single timestep.
            - list_of_params: list of parameters to include in the residual calculation. e.g. [pvs, pvs+prip, etc. (listed as col names)]
            - output_dir: directory to save the plot to.
        """
    # Step 1: Create a dict `lines`, where we will store mean residuals
    lines = {}
    for key in dir_dict[list(dir_dict.keys())[0]][0].keys():
        # create subdicts for every agent, using a default initial agents set
        _, agents = dir_dict[list(dir_dict.keys())[0]]
        lines[key] = [0]*len(dir_dict[list(dir_dict.keys())[0]][0]["HCVA"])


    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iteration, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents = data
        for key, cons_df in normalised_cons_sets.items():
            # cons_df will be a df for a cons. each row will correspond with a
            for j in range(0, len(cons_df.index)):
                # find the residuals for each of the cons in cons_dict_single
                # For every consensus, go through each agent, and find difference.
                temp_residuals = np.array([], dtype=float)
                for agent in normalised_agents:
                    temp_residual = np.abs(
                        agent[list_of_params].to_numpy() - cons_df.iloc[j][list_of_params].to_numpy()).sum()
                    temp_residuals = np.append(temp_residuals, [temp_residual])
                # Mean absolute difference
                mad = np.abs(np.subtract.outer(temp_residuals, temp_residuals)).mean()
                # Relative mean absolute difference
                rmad = mad / np.mean(temp_residuals)
                # Gini coefficient
                g = 0.5 * rmad
                lines[key][j] += g
    # Step 3: Given we have a total residual for each timestep and each method, divide by the number of cons
    for key in lines.keys():
        lines[key] = [iterator / len(dir_dict) for iterator in lines[key]]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    # Save list_of_points to a file
    with open(output_dir + title + "gini.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)
    return