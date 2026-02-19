import numpy as np
from datetime import datetime as dt
import csv
import random
import copy

def nonlinspace(start, stop, num):
    linear = np.linspace(0, 1, num)
    my_curvature = 1
    curve = 1 - np.exp(-my_curvature*linear)
    # Normalise between 0 and 1
    curve = curve/np.max(curve)
    #curve = curve*(stop - start-1) + start # don't minus 1 cause everything is between 0-1 for our use case.
    curve = curve * (stop - start) + start
    # Convert the curve from np to floats
    curve = [float(i) for i in curve]
    return curve

def generate_prips(agent_groups, curve_groups, n_principles):
    """Generates one preference for egalitarianism and returns as a dict of agents/prips"""
    prips = {}
    for curve_group in curve_groups:
        curve_values = curve_groups[curve_group][0]
        for agent in agent_groups:
            random_index = random.randint(0, len(curve_values)-1)
            prips[agent] = curve_values[random_index]
    return prips

def generate_ps(agent_groups, curve_groups, n_values):
    """Generates value preferences for n_values different values for a given number of agents"""
    value_preferences = {}
    """array([[ [uni, uni], [uni, ben], [uni, tra]],
            [ [ben, uni], [ben, ben], [ben, tra]],
            [ [tra, uni],  [tra,ben],  [tra, tra]]])"""
    # Get the first round of strengths
    for curve_group, agents in agent_groups.items():
        opposing_curve_group_index = len(curve_groups) - (1+list(curve_groups.keys()).index(curve_group))
        opposing_curve_group = list(curve_groups.keys())[opposing_curve_group_index]
        curve_values = curve_groups[curve_group][0]
        opposing_curve_values = curve_groups[opposing_curve_group][0]
        # For every agent
        for agent in agents:
            agent_strengths = []
            # Find the strength for half of values.
            agent_strengths = random.choices(curve_values, k=int(n_values/2))
            # Find the opposing strengths for the other half
            agent_strengths = agent_strengths + (random.choices(opposing_curve_values, k=int(n_values/2)))
            agent_strengths = np.array(agent_strengths)
            # Find agent preferences from the strength of preference for each value
            # For every value in the values list, compare it to every other value
            diff = agent_strengths[:, np.newaxis] - agent_strengths
            # Diff norm normalises the value preferences between 0 and 1
            diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
            value_preferences[agent] = copy.copy(diff_norm)
    return value_preferences

def generate_vas(agent_groups, curve_groups, n_values):
    """Generates action judgements for certain actions and preferences, returns as a dict of agents/vas"""
    action_judgements = {}
    # Should be fairly similar to principles?
    return action_judgements

def save_to_file(value_preferences, action_judgements, principle_prefs, agents_ids, n_values):
    now = dt.now().isoformat()
    principles_fn = now+"_PriP.csv"
    # Principles:
    with open(principles_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["country", "Egalitarian"]
        writer.writerow(header)
        # Rows
        for country in agents_ids:
            PriP = np.array(principle_prefs[country], dtype=float)
            row = [country, PriP]
            writer.writerow(row)

    # Values + Preferences + Action Judgements
    values_fn = now + "_PVS.csv"
    with open(values_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["country"]
        # P__vi__vj columns
        # TODO: Make this n_values automatically generate headers
        for vi in value_names:
            for vj in value_names:
                header.append(f"P__{vi}__{vj}")
        # VA__v__a columns
        for v in value_names:
            for a in action_names:
                header.append(f"VA__{v}__{a}")
        writer.writerow(header)
        # Rows
        for country in df[country_col_name].unique():
            P = np.array(value_preferences[country], dtype=float)
            VA = np.array(action_judgements[country], dtype=float)

            row = [country]
            for i in range(len(value_names)):
                for j in range(len(value_names)):
                    row.append(float(P[i, j]))

            for i in range(len(value_names)):
                for k in range(len(action_names)):
                    row.append(float(VA[i, k]))

            writer.writerow(row)
    return

if __name__ == "__main__":
    random.seed(10)
    # Define your curve of popularity for each input type.
    n_values = 4
    n_acts = 2
    n_agents = 30
    agent_ids = list(range(n_agents))

    # Each agent is going to randomly receive the following vals:
    #   One for principle preferences.
    #   16, one for every possible combination of value preferences (so 4 values will need 16 preferences).
    #   Two for action judgements (one per action)
    # However, we want to ensure there are clusters of agents. e.g. Agents who prefer A and B/C and D also prefer action 1/2.

    # Split the curve into different groups (extreme low, low, indifference, high, extreme high) (curve between 0/1)
    curve_groups = {"ex_low": [nonlinspace(0, 0.2, 10)], "low": [nonlinspace(0.2, 0.4, 10)],
                    "med": [nonlinspace(0.4,0.6, 10)], "high": [nonlinspace(0.6, 0.8, 10)],
                    "ex_high": [nonlinspace(0.8, 1, 10)]}
    print(curve_groups.items())

    # 1. Majority/Minority case with highly opposing views:
    # agent_groups splits the agents into aligned groups. These strengths will be for the first 50% of values.
    agent_groups = {"ex_low": agent_ids[:20], "ex_high": agent_ids[20:]}
    print(agent_groups.items())

    # Get PVSs and PriPs
    # These return Dicts in format {agent: [prefs]}
    value_preferences = generate_ps(agent_groups, curve_groups, n_values)
    action_judgements = generate_vas(agent_groups, curve_groups)
    principle_prefs = generate_prips(agent_groups, curve_groups)
    # Save to a csv
    save_to_file(value_preferences, action_judgements, principle_prefs, agent_ids, n_values)