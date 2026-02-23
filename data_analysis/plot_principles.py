
def mechanism_congruence(cons_df, agents_df):
    """Calculates the proportion of axioms that are satisfied by the consensus and the agents' preferred principle"""
    # Axioms can be IIA, Neutrality, Pareto Efficiency, etc.
    # 1 will mean the rule behaves the exact same as the agent's preferred principle
    axioms = ['IIA', 'Neutrality', 'Pareto Efficiency']
    cons_axioms = cons_df[axioms].sum(axis=1)
    agents_axioms = agents_df[axioms].sum(axis=1)
    return cons_axioms / agents_axioms

def procedural_residuals(cons_df, agents_df):
    """ Measures the disutility of using a specific ethical principle. Metric: U_total=U_outcome+y⋅U_procedural, where y
    is the distance the agent's principle is from the consensus principle.
    The difference between the utility of the outcome under the used rule vs. the utility of the outcome that
    would have occurred under the agent’s preferred rule. If the outcomes are the same, the residual is zero,
    implying "procedural indifference."
    """
    return cons_df['U_total'] - agents_df['U_total']

def stability_under_principle_voting():
    """ Is there another aggregation rule that a majority (or a specific coalition) of agents would prefer over the current one?
    If so, how many? Does a consensus have institutional drift? This is where agents are likely to contest the outcome."""


