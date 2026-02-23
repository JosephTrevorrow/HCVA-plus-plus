"""
This file will plot residuals given a consensus and a set of agents PVSs and PriPs. The gini index is also plotted
What makes up the residual can be set as an argument.
"""
import pandas as pd
import matplotlib.pyplot as plt

def gini_coefficent(cons_df, agents_df):
    """Calculates the Gini coefficient (Inequality of disappointment amongst agents)
    High total utility with high Gini means cons favours majority at expense of minority"""


def check_maximin_fairness(cons_df, agents_df, list_of_params):
    """Calculates the utility of the worst off agent in the society"""
    for cons in cons_df.iterrows():
        for agent in agents_df.iterrows():
            # For every col, match these two dfs and plot the residuals
            utils = (cons[1][list_of_params], agent[1][list_of_params] - cons[1][list_of_params])
            print(utils)


def calc_envy_freeness():
    """Calculates if an agent is envious of another consensus? Would they
    prefer if another consensus was chosen than the cons considered?"""

def plot_residuals(cons_df, agents_df, list_of_params, title):
    """Plots a residual bar chart given a list of parameters using the dataframe. Style will follow prev. work in the field.
    X Axis: Ps
    Y Axis: Residuals"""
    # For every cons in cons_df, plot the residuals over all agents in the agents df
    points = []
    for cons in cons_df.iterrows():
        for agent in agents_df.iterrows():
            # For every col, match these two dfs and plot the residuals
            points.append((cons[1][list_of_params], agent[1][list_of_params] - cons[1][list_of_params]))

    print(points)
    points = pd.DataFrame(points, columns=['cons', 'residual'])
    points.plot.bar(x='cons', y='residual', figsize=(10, 5), title=title)

if __name__ == "__main__":
    # Load in dataframe
    cons_df = pd.read_csv("COMPLETE_PERSONALS_2026....csv")
    agents_df = pd.read_csv("agents_df.csv")

    list_of_params = []
    title = "Pretty cold for a placehold"

    # Uses cons_df to unpack the values and actions lists, but cols for both dfs should be identical
    values_list, actions_list, principles_list = [], [], []
    if "P" in list_of_params:
        # Drop P from list_of_params
        list_of_params.remove("P")
        values_list = list([col for col in cons_df.columns if 'P__' in col])
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

    # Clean list_of_params
    # Remove all cols that have the same two values (P__Universalism__Universalism, P__Benevolence__Benevolence, etc.)
    for col in list_of_params:
        col_split = col.split("__")
        if len(col_split) == 3 and col_split[1] == col_split[2]:
            list_of_params.remove(col)
        else:
            # Not dropped, so drop the symmetrical col (P__A__B == P__B__A)
            symmetrical_col = "P__" + col_split[2] + "__" + col_split[1]
            if col in list_of_params:
                list_of_params.remove(col)

    print("len of list_of_params ", len(list_of_params))
    print(list_of_params)

    # Filter cons and agents_df considering list_of_params
    cons_df = cons_df[list_of_params]
    agents_df = agents_df[list_of_params]

    # Plot
    plot_residuals(cons_df, agents_df, list_of_params, title)