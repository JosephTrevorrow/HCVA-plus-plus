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

def plot_mean_residuals(dir_dict, list_of_params, title, output_dir):
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
    for key in dir_dict[list(dir_dict.keys())[0]][0].keys():
        # create an empty list to store the mean residuals for each timestep
        # find the number of timesteps using the length of normalised_cons_sets
        lines[key] = [0]*len(dir_dict[list(dir_dict.keys())[0]][0]["HCVA"])
    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iteration, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents = data
        for key, cons_df in normalised_cons_sets.items():
            # cons_df will be a df for a cons. each row will correspond with a
            for j in range(0,len(cons_df.index)):
            # find the residuals for each of the cons in cons_dict_single
            # For every consensus, go through each agent, and find difference.
                sum_of_residuals = 0
                temp_residual = np.abs(
                    normalised_agents[j][list_of_params].to_numpy() - cons_df.iloc[j][list_of_params].to_numpy()).sum()
                sum_of_residuals += temp_residual
                # Update the mean residual for this method and timestep
                lines[key][j] += sum_of_residuals

    # Step 3: Given we have a total residual for each timestep and each method, divide by the number of cons
    for key in lines.keys():
        lines[key] = [iteration/len(dir_dict) for iteration in lines[key]]

    # Step 4: Plot the lines and save
    """
    fig, ax = plt.subplots()

    for key, list_of_points in lines.items():
        #print("key is: ", key)
        #print("y is: ", list_of_points)
        #print("x is: ", x)
        y = list_of_points
        x = np.arange(len(y))
        ax.plot(x, y, label=key)
    ax.legend()
    output_dir = "HCVA-plus-plus/plots/" + output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    fig.savefig(output_dir + title + ".png", bbox_inches="tight")
    """

    # Save list_of_points to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    with open(output_dir + title + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)
    """
    # Write each line to a single csv (for use with creating latex plots
    for key, points in lines.items():
        with open(output_dir + title + "residuals_" + key + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "residual"])
            for timestep, residual in enumerate(points):
                row = [timestep, residual]
                writer.writerow(row)
    """
    return

def plot_violin_individuals(dir_dict, list_of_params, title, output_dir, difference):
    """Plots a violin graph, where each agent's residuals are plotted as a violin graph, and layered ontop of each other, per timestep.
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
    # Step 1: Create a dict `lines`, where we will store mean residuals for each method, for each timestep
    lines = {}
    for key in dir_dict[list(dir_dict.keys())[0]][0].keys():
        # create subdicts for every agent, using a default initial agents set
        _, agents = dir_dict[list(dir_dict.keys())[0]]
        lines[key] = {}
        for i in range (0, len(agents[0])):
            lines[key][i] = [0 for i in range(0, 50)]

    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iteration, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents = data
        for key, cons_df in normalised_cons_sets.items():
            # cons_df will be a df for a cons. each row will correspond with a
            for j in range(0,len(cons_df.index)):
                # find the residuals for each of the cons in cons_dict_single
                # For every consensus, go through each agent, and find difference.
                for timestep_id, agent in enumerate(normalised_agents):
                    temp_residual = np.abs(agent[list_of_params].to_numpy() - cons_df.iloc[j][list_of_params].to_numpy()).sum(axis=1)
                    if difference and timestep_id >0 and j >0:
                        previous_residual = np.abs(normalised_agents[timestep_id - 1][list_of_params].to_numpy() - cons_df.iloc[j-1][list_of_params].to_numpy()).sum(axis=1)
                    # Add this agent's residual data to a line for the agent
                    # Now add to lines for each agent
                    for i in range(0,len(temp_residual)):
                        if difference and timestep_id !=0 and j > 0:
                            lines[key][i][timestep_id]+= temp_residual[i] - previous_residual[i]
                        else:
                            lines[key][i][timestep_id]+=temp_residual[i]
    # Step 3: Given we have a total residual for each timestep and each method, divide by the number of cons to scale!
    for key in lines.keys():
        for i in range(0, len(lines[key])):
            lines[key][i] = [element / len(dir_dict) for element in lines[key][i]]
    """
    # Step 4: Plot the lines and save
    #fig, ax = plt.subplots()

    #for key, list_of_points in lines.items():
    #    y = list_of_points
    #    x = np.arange(len(y))
    #    ax.plot(x, y, label=key)
    #ax.legend()
    # creating figure and axes to
    # plot the image
    #fig, ax_list = plt.subplots(nrows=2,
                                   ncols=3,
                                   figsize=(20, 30),
                                   sharey=True)
    # plotting violin
    #for ax, key in zip(ax_list.flatten(), lines.keys()):
    #    ax.set_title(key)
    #    ax.set_ylabel('Observed values')
    #    for i in range(0, len(list(lines[key].keys()))):
    #        ax.violinplot(lines[key][i])
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    #fig.savefig(output_dir + title + ".png", bbox_inches="tight")

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
    # Step 1: Create a dict `lines`, where we will store mean residuals for each method, for each timestep
    lines = {}
    # for the normalised_cons_sets, first consensus dict, and their keys
    for key in dir_dict[list(dir_dict.keys())[0]][0].keys():
        # create an empty list to store the mean residuals for each timestep
        # find the number of timesteps using the length of normalised_cons_sets
        lines[key] = [0]*len(dir_dict[list(dir_dict.keys())[0]][0]["HCVA"])
    # Step 2: Iterate over every timestep (each [normalised_cons_sets, normalised_agents_df] in dir_dict)
    #   find the residuals for each of the cons, for each timestep, and add to lines
    for iteration, data in dir_dict.items():
        # Unpack
        normalised_cons_sets, normalised_agents = data
        for key, cons_df in normalised_cons_sets.items():
            # cons_df will be a df for a cons. each row will correspond with a
            for j in range(0,len(cons_df.index)):
                # find the residuals for each of the cons in cons_dict_single
                # For every consensus, go through each agent, and find difference.
                temp_residuals = np.array([], dtype=float)
                for agent in normalised_agents:
                    temp_residual = np.abs(agent[list_of_params].to_numpy() - cons_df.iloc[j][list_of_params].to_numpy()).sum()
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

    """
    # Step 4: Plot the lines and save
    fig, ax = plt.subplots()

    for key, list_of_points in lines.items():
        y = list_of_points
        x = np.arange(len(y))
        ax.plot(x, y, label=key)
    ax.legend()
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    #fig.savefig(output_dir + title + ".png", bbox_inches="tight")

    # Save list_of_points to a file
    with open(output_dir + title + "gini.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "points"])
        for key, points in lines.items():
            row = [key] + points
            writer.writerow(row)

    # Write each line to a single csv (for use with creating latex plots
    #for key, points in lines.items():
    #    with open(output_dir + title + "gini_" + key + ".csv", 'w') as f:
    #        writer = csv.writer(f)
    #        writer.writerow(["timestep", "gini"])
    #        for timestep, gini in enumerate(points):
    #            row = [timestep, gini]
    #            writer.writerow(row)
    return

"""
def procedural_residuals(cons_df, agents_df, principles_df):
    # NOT USED IN PAPER
     Measures the disutility of using a specific ethical principle. Metric: U_total=U_outcome+y⋅U_procedural, where y
    is the distance the agent's principle is from the consensus principle -> the distance is in terms of the outcome (PVS).
    The difference between the utility of the outcome under the used rule vs. the utility of the outcome that
    would have occurred under the agent’s preferred rule. If the outcomes are the same, the residual is zero,
    implying "procedural indifference."

    for cons in cons_df.iterrows():
        for agent in agents_df.iterrows():
            # Find the agent's utility with the cons


            # Find the agent's utility with their preferred cons
            agent_principle = principles_df

            cons_df.loc[cons[0], 'U_total'] = cons[1]['U_outcome'] + cons[1]['y'] * cons_principle
            agents_df.loc[agent[0], 'U_total'] = agent[1]['U_outcome'] + agent[1]['y'] * agent_principle


    return cons_df['U_total'] - agents_df['U_total']
"""