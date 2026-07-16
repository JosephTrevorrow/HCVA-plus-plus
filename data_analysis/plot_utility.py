import csv
import matplotlib as plt
import numpy as np
import os

def plot_total_utility(cons_sets, agents_df, list_of_params, filename, output_dir):
    """Find the total utility for all agents."""
    utilities = {}
    for key, cons_df in cons_sets.items():
        for cons_i in cons_df.iterrows():
            utility = 0
            for agent in agents_df.iterrows():
                temp_residual = cons_i[1][list_of_params] - agent[1][list_of_params]
                temp_residual = abs(temp_residual.sum())
                utility += temp_residual
            utilities[key] = utility
    ## Add to/Make a utilities csv file and save
    with open("plots/"+filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(utilities.keys())
        writer.writerow(utilities.values())
    return utilities

def plot_utility_over_time(consensus_list, agents_list, list_of_params, title, dir):
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
            sum_of_utilities = 0
            for agent in agents_single.iterrows():
                temp_utility = np.abs(agent[1][list_of_params].to_numpy() - cons_df[list_of_params].to_numpy()).sum()
                sum_of_utilities += temp_utility
            lines[key].append(sum_of_utilities)
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

def plot_pareto_efficiency(cons_df, agents_df, list_of_params):
    """NOT USED IN PAPER. Is there a cons that would make at least one agent better off
    without making another agent worse off?"""

    # Find the utilities for all cons, for all agents.
    utilities = {}
    for cons in cons_df.iterrows():
        temp_residuals = []
        for agent in agents_df.iterrows():
            temp_residual = cons[1][list_of_params] - agent[1][list_of_params]
            temp_residual= abs(temp_residual.sum())
            temp_residuals.append(temp_residual)
        utilities[cons[0]] = temp_residuals

    # Now compare the utilities between each other, seeing if there is ever a case where one cons has at least one
    #   agent that is better off, but never an agent worse off.
    for cons_name_i, utility_i in utilities.items():
        for cons_name_j, utility_j in utilities.items():
            betterOff = 0
            worseOff = 0
            for x in range(len(utility_i)):
                if utility_i[x] > utility_j[x]:
                    betterOff +=1
                elif utility_i[x] < utility_j[x]:
                    worseOff +=1
            if betterOff > 0 and worseOff == 0:
                print("There is a cons that would make at least one agent better off without making another agent worse off.")
                print("Compared ", cons_name_i, " and ", cons_name_j, ". ", betterOff, " agents are better off, ", worseOff, " agents are worse off.")