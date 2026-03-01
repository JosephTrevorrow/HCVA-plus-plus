def plot_pareto_efficiency(cons_df, agents_df, list_of_params):
    """Is there a cons that would make at least one agent better off
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

def plot_total_utility(cons_df, agents_df, list_of_params):
    """Find the total utility for all agents."""
    utilities = {}
    for cons_i in cons_df.iterrows():
        utility = 0
        for agent in agents_df.iterrows():
            temp_residual = cons_i[1][list_of_params] - agent[1][list_of_params]
            temp_residual = abs(temp_residual.sum())
            utility += temp_residual
        utilities[cons_i[0]] = utility
    print("Utilities are: ", utilities)
    return utilities
