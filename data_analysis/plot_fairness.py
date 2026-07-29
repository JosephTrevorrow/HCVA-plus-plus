"""
This file will plot residuals given a consensus and a set of agents PVSs and PriPs. The gini index is also plotted
What makes up the residual can be set as an argument.
"""
import csv
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    normalised = []
    for timestep in sets:
        normalised_sets = {}
        for key, df in timestep.items():
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
            normalised_sets[key] = copy.deepcopy(df)
        normalised.append(copy.deepcopy(normalised_sets))
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


def gini_coefficient(cons_sets, agents_df, list_of_params, filename, dir):
    """Calculates the Gini coefficient (Inequality of disappointment amongst agents)
    Low total utility with High Gini means cons favours majority at expense of minority (low because lower is better)"""
    ginis = {}
    for key, cons_df in cons_sets.items():
        for cons in cons_df.iterrows():
            temp_residuals = np.array([], dtype=float)
            for agent in agents_df.iterrows():
                # For every col, match these two dfs
                temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
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
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + dir
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

def plot_residuals(cons_sets, agents_df, list_of_params, title, dir):
    """Plots a residual bar chart given a list of parameters using the dataframe. Style will follow prev. work.
    This function makes a boxplot chart, where each plot is a method. This is for one experiment. If you are doing an
     experiment over time, the residuals can be plotted as a line graph, in the function below
    X Axis: Ps, Y Axis: Residuals
    The plot includes bars for every baseline method"""
    boxplots = {}
    metadata = []
    # For every consensus, go through each agent, and find difference.
    for key, cons_df in cons_sets.items():
        points = []
        for agent in agents_df.iterrows():
            # For every col, match these two dfs and find the sum of aboslute difference between all of the vals to compare.
            temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
            points.append(copy.copy(temp_residual))
        boxplots[key] = copy.copy(points)
        # Get the last element of boxplots (the one we just made), find its var, std, mean, etc. and save to dict with p as title
        metadata.append({'key': key,'mean': np.mean(points), 'std': np.std(points), 'var': np.var(points), 'min': np.min(points), 'max': np.max(points), 'points':copy.copy(points)})
    ## Make the boxplots
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(boxplots.values(), labels=boxplots.keys(), orientation='horizontal')
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir+title+".png", bbox_inches="tight")
    # Now make sure you save the boxplot data (mean/IQR/Whiskers) in a csv, with info for that run
    with open(output_dir+title+"residuals.csv", 'w') as f:
        header = ['key', 'mean', 'std', 'var', 'min', 'max', 'points']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(metadata)

def plot_residuals_over_time(consensus_list, agents_list, list_of_params, title, dir):
    """Plots a line graph showing the total residuals per method over time
    Where time=whatever parameter we varied"""

    # lines is a dict {ID: [t1_{sum of residuals}, t2_ etc. ]}, one list per line, one line per method.
    lines = {}
    for key in consensus_list[0].keys():
        lines[key] = []
    ## consensus_list shape: list[{dict}] [ {t1_hcva, t1_inf, t1_slm, etc.}, {t2_hcva, t2_inf, t2_slm, etc.}, etc.
    ## agents_list shape [t1_agents, t2_agents, etc.
    ## Get our data for each line:
    for cons_dict_single, agents_single in zip(consensus_list, agents_list):
        # find the residuals for each of the cons in cons_dict_single
        # For every consensus, go through each agent, and find difference.
        for key, cons_df in cons_dict_single.items():
            #points = []
            sum_of_residuals = 0
            for agent in agents_single.iterrows():
                temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
                sum_of_residuals += temp_residual
            lines[key].append(sum_of_residuals)
    # Make the line graph
    fig, ax = plt.subplots()

    for key, list_of_points in lines.items():
        y = list_of_points
        x = np.arange(len(y))
        ax.plot(x,y,label=key)
    ax.legend()
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir+title+".png", bbox_inches="tight")
    # Save list_of_points to a file
    with open(output_dir+title+"residuals.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)
    return

def plot_mean_residuals(dir_dict, list_of_params, title, output_dir, x):
    """Plots a line graph. Each line is the mean residual of a method over time. By time, we mean the varying of a parameter (e.g. increasing mu_p and mu_va).
        INPUTS:
        - dir_dict a dict of form {x : [normalised_cons_sets, normalised_agents_df]}, x=[0,100], x+=1
            - normalised_cons_sets is a list of dicts, [{hcva: cons, inf:cons}, {hcva: cons, inf: cons}, etc. ]
                - where each dict is all cons for a single timestep.
        - list_of_params: list of parameters to include in the residual calculation. e.g. [pvs, pvs+prip, etc. (listed as col names)]
        - output_dir: directory to save the plot to.
    """
    #print("x is: ", x)
    #print("len of x is: ", len(x))

    # Step 1: Create a dict `lines`, where we will store mean residuals for each method, for each timestep
    lines = {}
    # for the normalised_cons_sets, first consensus dict, and their keys
    for key in dir_dict[0][0][0].keys():
        # create an empty list to store the mean residuals for each timestep
        # find the number of timesteps using the length of normalised_cons_sets
        lines[key] = [0]*len(dir_dict[0][0])

    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iteration, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents_df = data
        # Do what we normally do with a single timestep:
        for i, (cons_dict_single, agents_single) in enumerate(zip(normalised_cons_sets, normalised_agents_df)):
            # find the residuals for each of the cons in cons_dict_single
            # For every consensus, go through each agent, and find difference.
            for key, cons_df in cons_dict_single.items():
                sum_of_residuals = 0
                for agent in agents_single.iterrows():
                    temp_residual = np.abs(
                        agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
                    sum_of_residuals += temp_residual
                # Update the mean residual for this method and timestep
                lines[key][i] += sum_of_residuals

    # Step 3: Given we have a total residual for each timestep and each method, divide by the number of cons
    for key in lines.keys():
        lines[key] = [iteration/len(dir_dict) for iteration in lines[key]]

    # Step 4: Plot the lines and save
    fig, ax = plt.subplots()

    for key, list_of_points in lines.items():
        #print("key is: ", key)
        #print("y is: ", list_of_points)
        #print("x is: ", x)
        y = list_of_points
        #x = np.arange(len(y))
        ax.plot(x, y, label=key)
    ax.legend()
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir + title + ".png", bbox_inches="tight")
    # Save list_of_points to a file
    with open(output_dir + title + "residuals.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)

    # Write each line to a single csv (for use with creating latex plots
    for key, points in lines.items():
        with open(output_dir + title + "residuals_" + key + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "residual"])
            for timestep, residual in enumerate(points):
                row = [timestep, residual]
                writer.writerow(row)
    return

def plot_mean_gini(dir_dict, list_of_params, title, output_dir, x):
    """Plots a line graph. Each line is the mean residual of a method over time. By time, we mean the varying of a parameter (e.g. increasing mu_p and mu_va).
            INPUTS:
            - dir_dict a dict of form {x : [normalised_cons_sets, normalised_agents_df]}, x=[0,100], x+=1
                - normalised_cons_sets is a list of dicts, [{hcva: cons, inf:cons}, {hcva: cons, inf: cons}, etc. ]
                    - where each dict is all cons for a single timestep.
            - list_of_params: list of parameters to include in the residual calculation. e.g. [pvs, pvs+prip, etc. (listed as col names)]
            - output_dir: directory to save the plot to.
        """
    # Step 1: Create a dict `lines`, where we will store mean residuals for each method, for each timestep
    lines = {}
    # for the normalised_cons_sets, first consensus dict, and their keys
    for key in dir_dict[0][0][0].keys():
        # create an empty list to store the mean residuals for each timestep
        # find the number of timesteps using the length of normalised_cons_sets
        lines[key] = [0] * len(dir_dict[0][0])

    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iterator, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents_df = data
        # Do what we normally do with a single timestep:
        for i, (cons_dict_single, agents_single) in enumerate(zip(normalised_cons_sets, normalised_agents_df)):
            # find the residuals for each of the cons in cons_dict_single
            # For every consensus, go through each agent, and find difference.
            for key, cons_df in cons_dict_single.items():
                temp_residuals = np.array([], dtype=float)
                for agent in agents_single.iterrows():
                    temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
                    temp_residuals = np.append(temp_residuals, [temp_residual])
                # Mean absolute difference
                mad = np.abs(np.subtract.outer(temp_residuals, temp_residuals)).mean()
                # Relative mean absolute difference
                rmad = mad / np.mean(temp_residuals)
                # Gini coefficient
                g = 0.5 * rmad
                lines[key][i] += g

    # Step 3: Given we have a total residual for each timestep and each method, divide by the number of cons
    for key in lines.keys():
        lines[key] = [iterator / len(dir_dict) for iterator in lines[key]]

    # Step 4: Plot the lines and save
    fig, ax = plt.subplots()

    for key, list_of_points in lines.items():
        y = list_of_points
        #x = np.arange(len(y))
        ax.plot(x, y, label=key)
    ax.legend()
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir + title + ".png", bbox_inches="tight")

    # Save list_of_points to a file
    with open(output_dir + title + "gini.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)

    # Write each line to a single csv (for use with creating latex plots
    for key, points in lines.items():
        with open(output_dir + title + "gini_" + key + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "gini"])
            for timestep, gini in enumerate(points):
                row = [timestep, gini]
                writer.writerow(row)
    return

def plot_gini_over_time(consensus_list, agents_list, list_of_params, title, dir):
    """Plots a line graph showing the gini coefficient over time
    Where time=whatever parameter we varied"""

    # lines is a dict {ID: [t1_{sum of residuals}, t2_ etc. ]}, one list per line, one line per method.
    ginis = {}
    for key in consensus_list[0].keys():
        ginis[key] = []
    ## consensus_list shape: list[{dict}] [ {t1_hcva, t1_inf, t1_slm, etc.}, {t2_hcva, t2_inf, t2_slm, etc.}, etc.
    ## agents_list shape [t1_agents, t2_agents, etc.
    ## Get our data for each line:
    for cons_dict_single, agents_single in zip(consensus_list, agents_list):
        # find the residuals for each of the cons in cons_dict_single
        # For every consensus, go through each agent, and find difference.
        for key, cons_df in cons_dict_single.items():
            # points = []
            sum_of_residuals = 0
            temp_residuals = np.array([], dtype=float)
            for agent in agents_single.iterrows():
                # For every col, match these two dfs
                temp_residual = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
                temp_residuals = np.append(temp_residuals, [temp_residual])
            # Mean absolute difference
            mad = np.abs(np.subtract.outer(temp_residuals, temp_residuals)).mean()
            if np.mean(temp_residuals) == 0:
                print("WARNING: Mean of residuals is 0.")
            # Relative mean absolute difference
            rmad = mad / np.mean(temp_residuals)
            # Gini coefficient
            g = 0.5 * rmad
            ginis[key].append(copy.copy(g))
    # Make the line graph
    fig, ax = plt.subplots()

    for key, list_of_points in ginis.items():
        y = list_of_points
        x = np.arange(len(y))
        ax.plot(x, y, label=key)
    ax.legend()
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/plots/" + dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir+title + ".png", bbox_inches="tight")
    # Save list_of_points to a file
    with open(output_dir+title + "gini.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in ginis.items():
            row = [key] + points
            writer.writerow(row)
    return

def check_maximin_fairness(cons_df, agents_df, list_of_params):
    """NOT USED IN PAPER. Calculates the utility of the worst off agent in the society"""
    worst_offs = {}
    for cons in cons_df.iterrows():
        temp_residuals = []
        for agent in agents_df.iterrows():
            # For every col, match these two dfs
            temp_residual = cons[1][list_of_params] - agent[1][list_of_params]
            temp_residual = abs(temp_residual.sum())
            temp_residuals.append(copy.copy(temp_residual))
        max_dist = max(temp_residuals)
        print("Worst Welfare is: ", max_dist)
        worst_offs[cons[0]] = max_dist
    return worst_offs

def calc_envy_freeness(cons_df, agents_df, list_of_params):
    """NOT USED IN PAPER. Calculates if an agent is envious of another consensus? Would they
    prefer if another consensus was chosen than the cons considered?"""
    residuals = {}
    for cons in cons_df.iterrows():
        temp_residuals = np.array([], dtype=float)
        for agent in agents_df.iterrows():
            # For every col, match these two dfs
            temp_residual = cons[1][list_of_params] - agent[1][list_of_params]
            temp_residual = abs(temp_residual.sum())
            temp_residuals = np.append(temp_residuals, [temp_residual])
        residuals[cons[0]] = temp_residuals
    # Because all the residuals are in order, how many agents have a better (lower) residual on any other given consensus?
    envy_count = {}
    for cons_i, residual_set_i in residuals.items():
        for cons_j, residual_set_j in residuals.items():
            num_envious = 0
            diffs = residual_set_i - residual_set_j
            for diff in diffs:
                if diff < 0:
                    num_envious += 1
        print("Number envious in ", cons_i, " is: ", num_envious, ".")
        envy_count[cons_i] = num_envious
    return envy_count