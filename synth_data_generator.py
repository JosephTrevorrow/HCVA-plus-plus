import numpy as np
from PIL.ImageOps import scale
from scipy.stats import truncnorm
from datetime import datetime as dt
import csv
import random
import copy
from datetime import date
from lp_regression.solve import *

from collections import defaultdict

## SEED USED IN ALL EXP.
seed = 4832095623890
rng = np.random.default_rng(seed)

def generate_prips(agent_ids, ps, vas, pvs_filename, n_values, n_actions, corr, scale_prip, sigma_noise):
    """Generates a PriP set for agents by considering the PVS and alignment."""
    P_list, J_list, w, country_dict = FormalisationObjects(filename=pvs_filename, delimiter=',', weights=0,
                                                           n_values=n_values, n_actions=n_actions)
    pvs_df = pd.read_csv(pvs_filename)
    values_list = list([col for col in pvs_df.columns if 'P__' in col])
    actions_list = list([col for col in pvs_df.columns if 'VA__' in col])
    ## INF
    _, _, cons_pref_inf = aggregate_inf(P_list, J_list, w, np.inf, True)
    _, _, cons_act_inf = aggregate_inf(P_list, J_list, w, np.inf, False)
    cons_act_inf = cons_act_inf[:len(cons_act_inf) // 2]
    ## 1
    _, _, cons_pref_1 = aggregate_one(P_list, J_list, w, 1, True)
    _, _, cons_act_1 = aggregate_one(P_list, J_list, w, 1, False)
    cons_act_1 = cons_act_1[:len(cons_act_1) // 2]
    # Now we have the consensus for complete util and complete egal, we can see which values of 1 promote
    # which values, and which values of inf promote the others.
    # So now for each agent, find their residual to inf and 1. Whichever is smaller, sample from a norm dist relevant to them
    # As the stored PriP is egalitarian, we say if inf is closer, then we centre on 1, if 1 is closer, we centre on 0.
    prips = {}
    for agent in agent_ids:
        inf_diff = np.abs((cons_pref_inf - ps[agent].flatten()) + (cons_act_inf - vas[agent].flatten())).sum()
        ones_diff = np.abs((cons_pref_1 - ps[agent].flatten()) + (cons_act_1 - vas[agent].flatten())).sum()
        if inf_diff > ones_diff:
            # if we want egal to be closer to 1
            # loc and scale are just mean and std for normal.
            # Halfnorm is at mu and points UP. so we start with mu = 1, scale = 0.08.
            #   Therefore the range is between [0.5,1], with 0.5 being high prob
            #   To truncate this, you have to normalise this. truncnorm expects the trunc points to be stds away from the mean.
            loc = 1-corr
            a_trunc = 0.5
            b_trunc = 1
            a, b = (a_trunc - loc) / scale_prip, (b_trunc - loc) / scale_prip
            prip = truncnorm.rvs(a, b, loc=loc, scale=scale_prip, size=1, random_state=rng)
        else:
            loc = 0+corr
            a_trunc = 0
            b_trunc = 0.5
            a, b = (a_trunc - loc) / scale_prip, (b_trunc - loc) / scale_prip
            prip = truncnorm.rvs(a, b, loc=loc, scale=scale_prip, size=1, random_state=rng)
        # add some random noise according to sigma_noise
        prip += rng.normal(0, sigma_noise, size=1)
        np.clip(prip, -1, 1, out=prip)
        prips[agent] = prip[0]
    return prips

def generate_ps(agent_ids, n_values, mu=0.75, group_ratio=0.5):
    agent_ps = defaultdict(lambda: np.empty(shape=(1,n_values)))
    global rng
    # Assign values to groups
    values_list = [i for i in range(n_values)]
    num_of_vals = rng.integers(1, len(values_list), size=1)
    strong_vals_group_a = rng.choice(values_list, size=num_of_vals, replace=False)
    # Assign agents to groups
    group_a = rng.integers(1, len(agent_ids), size=int((len(agent_ids)*group_ratio)))
    # Find strengths for each agent, considering their group and the values that group holds.
    for agent in agent_ids:
        if agent in group_a:
            samples_a= rng.normal(mu, 0.08, len(strong_vals_group_a))
            samples_b = rng.normal(mu-0.5, 0.08, len(values_list)-len(strong_vals_group_a))
        else:
            samples_a = rng.normal(mu-0.5, 0.08, len(strong_vals_group_a))
            samples_b = rng.normal(mu, 0.08, len(values_list)-len(strong_vals_group_a))
        # Save these samples in the right order
        for k in range(0, n_values):
            if k in strong_vals_group_a:
                # pop
                last, samples_a = samples_a[-1], samples_a[:-1]
                # This changes the default
                agent_ps[agent][0][k] = copy.deepcopy(last)
            else:
                last, samples_b = samples_b[-1], samples_b[:-1]
                agent_ps[agent][0][k] = copy.deepcopy(last)
    # get these values to be preferences
    for agent, agent_values in agent_ps.items():
        # for every value in the values list, compare it to every other value
        print("AGENT: ", agent)
        print("agent_values: ", agent_values)
        diff = agent_values[0][:, np.newaxis] - agent_values
        print("diff: ",diff)
        if np.max(diff) - np.min(diff) == 0:
            diff_norm = np.full(diff.shape, 0.5)
        else:
            diff_norm = (diff- np.min(diff)) / (np.max(diff) - np.min(diff))
        np.clip(diff_norm, 0, 1, out=diff_norm)
        agent_ps[agent] = copy.deepcopy(diff_norm)
    return agent_ps

def generate_vas(agent_ids, n_values, n_actions, va_p):
    """Generates action judgements for certain actions and preferences, returns as a dict of agents/vas"""
    # Assign actions to values
    global rng
    values_list = [i for i in range(n_values)]
    actions_list = [j for j in range(n_actions)]
    agent_vas = defaultdict(lambda: np.empty(shape=(0,n_values)))

    for _ in actions_list:
        num_of_vals = rng.integers(1, len(values_list), size=1)
        promoted_values = rng.choice(values_list, size=num_of_vals, replace=False)
        # Now we have the values the action promotes, take those, and find action judgements for every
        # value.
        for agent in agent_ids:
            va = np.array([])
            for value in values_list:
                if value in promoted_values:
                    va_temp = rng.normal()
                else:
                    va_temp = rng.normal()-1
                # Add a bit of noise and add to va
                noise = rng.normal(0, va_p, size=1)
                va_temp = va_temp+noise
                va = np.append(va, copy.deepcopy(va_temp))
            np.clip(va, -1, 1, out=va)
            agent_vas[agent] = np.append(agent_vas[agent], [copy.deepcopy(va)], axis=0)
    return agent_vas

def save_pvs(ps, vas, agents_ids, n_values, n_acts, filename):
    # Values + Preferences + Action Judgements
    now = str(date.today())
    values_fn = now + filename + "_PVS.csv"
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
            ## Is there something fishy with these two lines?
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
    return values_fn

def save_prips(prips, agents_ids, filename):
    now = str(date.today())
    principles_fn = now+filename+"_PriP.csv"
    # Principles:
    with open(principles_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["country", "Egalitarian"]
        writer.writerow(header)
        # Rows
        for country in agents_ids:
            PriP = prips[country]
            row = [country, PriP]
            writer.writerow(row)
    return

def generate(n_values, n_actions, n_agents, pvs_prip=0.3, va_p=0.8, mu_p=0.75, corr_prip=0.5, scale_prip=0.08, pvs_filename="default", prip_filename="default"):
    """Generates a synthetic data distribution.
    - n_values: number of values
    - n_actions: number of actions
    - n_agents: number of agents
    - pvs_prip: The amount of noise to add to the PriP values.
    - va_p: The amount of noise to add to the VA_p values.
    - mu_p: The mean of the normal dist. for p vals
    - corr_prip: The amount of change in PriP value correlation. 0 means normal distribution is centred on 1 or 0, 0.05 means
        distribution is centred on 0.95 or 0.05.
    - sigma_p: The std of the normal dist. for p vals
    - sigma_prip: the std of the normal dist. for PriP vals
    """
    agent_ids = list(range(n_agents))

    ## Step 1: Sample from a normal distribution to find the strength of values for every agent - Generating P strength.
    # Then, convert these strengths to pairwise preferences
    ps = generate_ps(agent_ids, n_values, mu_p, group_ratio=0.5)

    ## Step 1.2: Given the number of actions and values, randomly assign an action 1 or more values it promotes.
    # The remaining values are demoted. This can always be random.
    vas = generate_vas(agent_ids, n_values, n_actions, va_p)

    ## Step 1.3 Given the ps and vas, save the pvs to a csv.
    pvs_dir = save_pvs(ps, vas, agent_ids, n_values, n_actions, pvs_filename)

    ## Step 3: For every agent's PVS, we can use the value aggregation code to find aggregations of 1 and inf. We find the consensus that minimises
    # the agent's residual. This consensus (when converted back to a preference) is the mean point of the normal distribution.
    prips = generate_prips(agent_ids, ps, vas, pvs_dir, n_values, n_actions, corr_prip, scale_prip, pvs_prip)
    ## Step 3.1: Save prips to a csv.
    save_prips(prips, agent_ids, prip_filename)
    return ps, vas, prips, agent_ids

if __name__ == "__main__":
# DEFAULTS: def generate(n_values, n_actions, n_agents, pvs_prip=0.3, va_p=0.8, mu_p=0.75, mu_prip=0.5,
# scale_prip=0, pvs_filename="default", prip_filename="default"):
    ## MAJ/MIN SPLIT
    generate(n_values=3, n_actions=3, n_agents=12, pvs_prip=0, va_p=0.3, mu_p=0.75, corr_prip=0, scale_prip=0.08, pvs_filename="test", prip_filename="test")

    ## EXTREME PVSs

    ## PriP CORR

    ## PriP NOISE

    ## NUM AGENTS

    ## NUM VALUES

    ## NUM ACTIONS
