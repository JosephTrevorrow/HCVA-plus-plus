"""
This file will plot residuals given a consensus and a set of agents PVSs and PriPs. The gini index is also plotted
What makes up the residual can be set as an argument.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def gini_coefficient(cons_df, agents_df, list_of_params):
    """Calculates the Gini coefficient (Inequality of disappointment amongst agents)
    Low total utility with High Gini means cons favours majority at expense of minority (low because lower is better)"""
    ginis = {}
    for cons in cons_df.iterrows():
        temp_residuals = np.array([], dtype=float)
        for agent in agents_df.iterrows():
            # For every col, match these two dfs
            temp_residual = cons[1][list_of_params] - agent[1][list_of_params]
            temp_residual = abs(temp_residual.sum())
            temp_residuals = np.append(temp_residuals, [temp_residual])
        print("Temp residuals are: ", temp_residuals)
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(temp_residuals, temp_residuals)).mean()
        # Relative mean absolute difference
        rmad = mad / np.mean(temp_residuals)
        # Gini coefficient
        g = 0.5 * rmad
        print("Gini coefficient is: ", g)
        ginis[cons[0]] = g
    return ginis

def check_maximin_fairness(cons_df, agents_df, list_of_params):
    """Calculates the utility of the worst off agent in the society"""
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
    """Calculates if an agent is envious of another consensus? Would they
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

def plot_residuals(cons_df, agents_df, list_of_params, title):
    """Plots a residual bar chart given a list of parameters using the dataframe. Style will follow prev. work.
    X Axis: Ps, Y Axis: Residuals"""
    # For every cons in cons_df, plot the residuals over all agents in the agents df

    to_plot_df = pd.DataFrame()
    for cons in cons_df.iterrows():
        points = []
        for agent in agents_df.iterrows():
            # For every col, match these two dfs and plot the residuals
            temp_residual = cons[1][list_of_params] - agent[1][list_of_params]
            temp_residual = abs(temp_residual.sum())
            points.append(copy.copy(temp_residual))
        to_plot_df[cons[0]] = points

    to_plot_df.boxplot(patch_artist=True, boxprops=dict(facecolor="#b7e4c7"))
    plt.title(title, fontsize=14)
    plt.ylabel("Value", fontsize=11)
    plt.tick_params(axis="x", rotation=90)
    plt.ylim(0, 0.5)
    plt.grid(alpha=0.25)
    plt.savefig("residuals.png", bbox_inches="tight")