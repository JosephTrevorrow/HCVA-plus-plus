def plot_pareto_efficiency(cons_df, agents_df, list_of_params):
    """Is there a cons that would make at least one agent better off
    without making another agent worse off?"""
    # Compare cons with cons, double for loop
    for cons_i in cons_df.iterrows():
        for cons_j in cons_df.iterrows():
            # Given these two cons, find if there is any agent that would be worse off
            #   if that consensus was chosen
            for agent in agents_df.iterrows():
                temp_residual_i = cons_i[1][list_of_params] - agent[1][list_of_params]
                temp_residual_i = abs(temp_residual_i.sum())
                temp_residual_j = cons_j[1][list_of_params] - agent[1][list_of_params]
                temp_residual_j = abs(temp_residual_j.sum())
                if temp_residual_i == temp_residual_j:
                    print("Consensus ", cons_i[0], " and ", cons_j[0], " are equally good.")
                elif temp_residual_i > temp_residual_j:
                    print("Consensus ", cons_i[0], " is better than consensus ", cons_j[0], ".")
                else:
                    print("Consensus ", cons_j[0], " is better than consensus ", cons_i[0], ".")
        print("New cons J")
    print("New cons I")


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
