import scipy

def calc_polarisation_index(x, y):
    """Is cons mildly disliked by everyone (low variance, high residual) or
    is there a bimodal split (50% LOVE, 50% HATE)"""
    return (x - y).mean()

def calc_rank_corr(x, y):
    """ Compare ranked PVS for each agent against cons. Describes how well ordinal preferences are preserved
    regardless of cardinal magnitude"""
    return scipy.stats.kendalltau(x, y)[0]

def calc_value_sensitivity(x, y):
    """If promotions change slightly, does the final decision change? This is not preferable, shows less reliable system"""
    # Use all cons here in -range arg to show this.