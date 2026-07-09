import numpy as np
from datetime import datetime as dt
import csv
import random
import copy

seed = 4832095623890
rng = np.random.default_rng(seed)

def generate_prips(agent_groups, curve_groups):
    """Generates one preference for egalitarianism and returns as a dict of agents/prips.
    Note that there is no alignment here with PVSs, so this should be considered manually."""
    prips = {}
    for curve_group, agents in agent_groups.items():
        curve_values = curve_groups[curve_group][0]
        for agent in agents:
            random_index = random.randint(0, len(curve_values)-1)
            prips[agent] = curve_values[random_index]
    print(prips)
    return prips

def generate_ps(agent_ids, n_values, mu, sigma):
    agent_ps = {}
    global rng
    # Find strengths
    for agent in agent_ids:
        # Sample
        samples = rng.normal(mu, sigma, n_values)
        agent_ps[agent] = samples
    # get these values to be preferences
    for agent, agent_values in agent_ps.items():
        #for every value in the values list, compare it to every other value
        diff = agent_values[:, np.newaxis] - agent_values
        if np.max(diff) - np.min(diff) == 0:
            diff_norm = np.full(diff.shape, 0.5)
        else:
            diff_norm = (diff- np.min(diff)) / (np.max(diff) - np.min(diff))
        agent_ps[agent] = copy.copy(diff_norm)
    return agent_ps

def generate_vas(agent_ids, n_values, n_actions, mu, sigma, agent_ps):
    """Generates action judgements for certain actions and preferences, returns as a dict of agents/vas"""
    # Assign actions to values
    global rng
    values_list = [i for i in range(n_values)]
    actions_list = [j for j in range(n_actions)]
    default = np.empty((0, len(values_list)))
    agent_vas = dict.fromkeys(agent_ids, default)
    for action in actions_list:
        num_of_vals = rng.integers(1, len(values_list), size=1)
        promoted_values = rng.choice(values_list, size=num_of_vals, replace=False)
        print("I can promote ", num_of_vals, ". My promoted vals are: ", promoted_values)
        # Now we have the values the action promotes, take those, and find action judgements for every
        # value.
        va = []
        for agent in agent_ids:
            for value in values_list:
                if value in promoted_values:
                    va.append(rng.random())
                else:
                    va.append(rng.random()-1)
            agent_vas[agent] = np.append(agent_vas[agent], [copy.copy(va)], axis=0)
    return agent_vas

def save_to_file(ps, vas, prips, agents_ids, n_values, n_acts):
    now = dt.now().isoformat()
    """
    principles_fn = now+"_PriP.csv"
    # Principles:
    with open(principles_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["country", "Egalitarian"]
        writer.writerow(header)
        # Rows
        for country in agents_ids:
            PriP = principle_prefs[country]
            row = [country, PriP]
            writer.writerow(row)
    """
    # Values + Preferences + Action Judgements
    values_fn = now + "_PVS.csv"
    with open(values_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["country"]
        # P__vi__vj columns
        for vi in range(n_values):
            for vj in range(n_values):
                header.append(f"P__{vi}__{vj}")
        # VA__v__a columns
        for v in range(n_values):
            for a in range(n_acts):
                header.append(f"VA__{v}__{a}")
        writer.writerow(header)
        # Rows
        for agent in agents_ids:
            P = np.array(ps[agent], dtype=float)
            VA = np.array(vas[agent], dtype=float)

            row = [agent]
            for i in range(n_values):
                for j in range(n_values):
                    row.append(float(P[i, j]))

            for i in range(n_values):
                for k in range(n_acts):
                    row.append(float(VA[i, k]))

            writer.writerow(row)
    return

def generate(n_values=4, n_actions=2, n_agents=30, pvs_prip=0.3, va_p=0.8, mu_p=0.7, sigma_p=0.1):
    agent_ids = list(range(n_agents))

    ## Step 1: Sample from a normal distribution to find the strength of values for every agent - Generating P strength.
    # Then, convert these strengths to pairwise preferences
    ps = generate_ps(agent_ids, n_values, mu_p, sigma_p)
    print("cor blimey! some peas!")
    print(ps[0])

    ## Step 2: Given the number of actions and values, randomly assign an action 1 or more values it promotes.
    # The remaining values are demoted. This can always be random.
    vas = generate_vas(agent_ids, n_values, n_actions, mu_p, sigma_p, ps)
    print("Agent VAs!")
    print(vas[0])

    ## Step 3: For every agent's new value system, we can use the value aggregation code to find a range of aggreagtions. We find the consensus that minimises
    # the agent's residual. This consensus (when converted back to a preference) is the mean point of the normal distribution.
    prips = None

    return ps, vas, prips, agent_ids

if __name__ == "__main__":
    ps, vas, prips, agent_ids = generate(n_values=2, n_actions=2, n_agents=1, pvs_prip=0.3, va_p=0.8, mu_p=0, sigma_p=0.1)
    # Save to CSV
    save_to_file(ps, vas, prips, agent_ids, n_values=2, n_acts=2)