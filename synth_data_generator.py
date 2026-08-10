import numpy as np
from PIL.ImageOps import scale
from scipy.stats import truncnorm
from datetime import datetime as dt
import csv
import random
import copy
from datetime import date
from lp_regression.solve import *
import os

from collections import defaultdict

## SEED USED IN ALL EXP.
seed = 4832095623890
rng = np.random.default_rng(seed)

def generate_prips(agent_ids, ps, vas, pvs_filename, n_values, n_actions, sigma_noise):
    """Generates a PriP set for agents by considering the PVS and alignment."""
    P_list, J_list, w, country_dict = FormalisationObjects(filename=pvs_filename, delimiter=',', weights=0,
                                                           n_values=n_values, n_actions=n_actions)
    ## INF
    _, _, cons_pref_inf = aggregate_inf(P_list, J_list, w, np.inf, True)
    _, _, cons_act_inf = aggregate_inf(P_list, J_list, w, np.inf, False)
    cons_act_inf = cons_act_inf[:len(cons_act_inf) // 2]
    ## 1
    _, _, cons_pref_1 = aggregate_one(P_list, J_list, w, 1, True)
    _, _, cons_act_1 = aggregate_one(P_list, J_list, w, 1, False)
    cons_act_1 = cons_act_1[:len(cons_act_1) // 2]
    # For each agent, find their residual to the egalitarian (inf) and utilitarian (1) consensus.
    # Minimising strategy: assign the PrIP whose consensus is closest to the agent's own preferences.
    # If the egalitarian consensus is closer (inf_diff < ones_diff), centre PrIP on 1 (high egalitarian).
    # If the utilitarian consensus is closer, centre PrIP on 0.
    prips = {}
    for agent in agent_ids:
        vas_flat = vas[agent].T.flatten()  # transpose to value-major order to match consensus ordering
        inf_diff = np.abs(cons_pref_inf - ps[agent].flatten()).sum() + np.abs(cons_act_inf - vas_flat).sum()
        ones_diff = np.abs(cons_pref_1 - ps[agent].flatten()).sum() + np.abs(cons_act_1 - vas_flat).sum()
        if inf_diff < ones_diff:
            # if we want egal to be closer to 1
            # loc and scale are just mean and std for normal.
            # We start with mu = p, scale = 0.08.
            #   To truncate this, you have to normalise this. truncnorm expects the trunc points to be stds away from the mean, rather than points on the x-axis
            loc = 1
            sigma_prip = 0.25
            a_trunc = 0.5
            b_trunc = 1
            a, b = (a_trunc - loc) / sigma_prip, (b_trunc - loc) / sigma_prip
            prip = truncnorm.rvs(a, b, loc=loc, scale=sigma_prip, size=1, random_state=rng)
        else:
            loc = 0
            sigma_prip = 0.25
            a_trunc = 0
            b_trunc = 0.5
            a, b = (a_trunc - loc) / sigma_prip, (b_trunc - loc) / sigma_prip
            prip = truncnorm.rvs(a, b, loc=loc, scale=sigma_prip, size=1, random_state=rng)
        # add some random noise according to sigma_noise, and clip the noise
        prip += rng.normal(0, sigma_noise, size=1)
        np.clip(prip, 0, 1, out=prip)
        prips[agent] = prip[0]
    return prips

def generate_ps(agent_ids, n_values, mu=0.75, p_group_factor=0.5):
    agent_ps = defaultdict(lambda: np.empty(shape=(1,n_values)))
    # Assign values to groups
    values_list = [i for i in range(n_values)]
    num_of_vals = rng.integers(1, len(values_list))
    strong_vals_group_a = rng.choice(values_list, size=int(num_of_vals), replace=False)
    # Assign agents to groups
    group_a = rng.choice(a=np.arange(len(agent_ids)), size=int((len(agent_ids)*p_group_factor)), replace=False)
    # Find strengths for each agent, considering their group and the values that group holds.
    for agent in agent_ids:
        if agent in group_a:
            samples_a= rng.normal(mu, 0.05, len(strong_vals_group_a))
            samples_b = rng.normal(mu-0.7, 0.05, len(values_list)-len(strong_vals_group_a))
        else:
            samples_a = rng.normal(mu-0.7, 0.05, len(strong_vals_group_a))
            samples_b = rng.normal(mu, 0.05, len(values_list)-len(strong_vals_group_a))
        # Save these samples in the right order
        for k in range(0, n_values):
            if k in strong_vals_group_a:
                # pop
                last, samples_a = samples_a[-1], samples_a[:-1]
                agent_ps[agent][0][k] = copy.deepcopy(last)
            else:
                last, samples_b = samples_b[-1], samples_b[:-1]
                agent_ps[agent][0][k] = copy.deepcopy(last)
    # get these values to be preferences
    for agent, agent_values in agent_ps.items():
        # for every value in the values list, compare it to every other value
        diff = agent_values[0][:, np.newaxis] - agent_values
        if np.max(diff) - np.min(diff) == 0:
            diff_norm = np.full(diff.shape, 0.5)
        else:
            diff_norm = (diff- np.min(diff)) / (np.max(diff) - np.min(diff))
        np.clip(diff_norm, 0, 1, out=diff_norm)
        agent_ps[agent] = copy.deepcopy(diff_norm)
    return agent_ps

def generate_vas(agent_ids, n_values, n_actions, va_p, va_mu):
    """Generates action judgements for certain actions and preferences, returns as a dict of agents/vas"""
    # Assign actions to values
    values_list = [i for i in range(n_values)]
    actions_list = [j for j in range(n_actions)]
    agent_vas = defaultdict(lambda: np.empty(shape=(0,n_values)))

    for _ in actions_list:
        num_of_vals = rng.integers(1, len(values_list)) #Having size be None, the default, means that a scalar will be returned
        promoted_values = rng.choice(values_list, size=int(num_of_vals), replace=False)
        # Now we have the values the action promotes, take those, and find action judgements for every
        # value.
        for agent in agent_ids:
            va = np.array([])
            for value in values_list:
                if value in promoted_values:
                    # Promoted, so must be closer to 1
                    va_temp = rng.normal(va_mu, 0.05)
                else:
                    # Demoted, so must be closer to -1
                    va_temp = rng.normal(-va_mu, 0.05)
                # Add a bit of noise and add to va
                noise = rng.normal(0, va_p, size=1)
                va_temp = va_temp+noise
                va = np.append(va, copy.deepcopy(va_temp))
            np.clip(va, -1, 1, out=va)
            agent_vas[agent] = np.append(agent_vas[agent], [copy.deepcopy(va)], axis=0)
    return agent_vas

def save_pvs(ps, vas, agents_ids, n_values, n_acts, filename, output_dir):
    # Values + Preferences + Action Judgements
    now = str(date.today())
    values_fn = now + "_"+filename + "_PVS.csv"
    print("The output output_dir is: ", output_dir, " and the filename is: ", values_fn, "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    with open(output_dir+values_fn, 'w', newline='') as csvfile:
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
            # Ensure n_values and n_acts are being used the same when writing columns. Value first is how ESS data is constructed, so we copy this.
            for i in range(n_values):
                for k in range(n_acts):
                    # Generate vas generates vas action major, although we want to store it value major.
                    row.append(float(VA[k, i]))
            writer.writerow(row)
    return output_dir+values_fn

def save_prips(prips, agents_ids, filename, output_dir):
    now = str(date.today())
    principles_fn = now+"_"+filename+"_PriP.csv"
    print("The output output_dir is: ", output_dir, " and the filename is: ", filename, "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    else:
        print("Directory already exists: " + output_dir)
    with open(output_dir+principles_fn, 'w', newline='') as csvfile:
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

def generate(n_values, n_actions, n_agents, pvs_prip=0.3, va_p=0.3,p_group_factor=0.5,
             mu_p=0.75, va_mu=0.75,  sigma_prip=0.08, pvs_filename="default", prip_filename="default", pvs_output_dir="", prip_output_dir=""):
    """Generates a synthetic data distribution.
    - n_values: number of values
    - n_actions: number of actions
    - n_agents: number of agents
    - pvs_prip: The amount of noise to add to the PriP values.
    - va_p: The amount of noise to add to the VA_p values.
    - mu_p: The mean of the normal dist. for p vals
    - sigma_p: The std of the normal dist. for p vals
    - p_group_factor: The percentage of agents assigned to a group.
    """
    agent_ids = list(range(n_agents))

    ## Step 1: Sample from a normal distribution to find the strength of values for every agent - Generating P strength.
    # Then, convert these strengths to pairwise preferences
    ps = generate_ps(agent_ids, n_values, mu_p, p_group_factor)
    ## Step 1.2: Given the number of actions and values, randomly assign an action 1 or more values it promotes.
    # The remaining values are demoted. This can always be random.
    vas = generate_vas(agent_ids, n_values, n_actions, va_p, va_mu)
    ## Step 1.3 Given the ps and vas, save the pvs to a csv.
    pvs_dir = save_pvs(ps, vas, agent_ids, n_values, n_actions, pvs_filename, pvs_output_dir)

    ## Step 3: For every agent's PVS, we can use the value aggregation code to find aggregations of 1 and inf. We find the consensus that minimises
    # the agent's residual. This consensus (when converted back to a preference) is the mean point of the normal distribution.
    prips = generate_prips(agent_ids, ps, vas, pvs_dir, n_values, n_actions, pvs_prip)
    ## Step 3.1: Save prips to a csv.
    save_prips(prips, agent_ids, prip_filename, prip_output_dir)
    return ps, vas, prips, agent_ids

if __name__ == "__main__":
    output_dir = "value_systems/Synthetic/"
    for run in range(500):
        ### Initial: agents, values, actions
        for ag in range(2, 30,1):
            generate(n_values=4, n_actions=2, n_agents=ag, pvs_prip=0.3, va_p=0.3, p_group_factor=0.5,
                    mu_p=0.85, va_mu=0.85, pvs_filename="vary_agents_"+str(ag), prip_filename="vary_agents_"+str(ag), pvs_output_dir=output_dir+"vary_agents/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_agents/PriP/"+str(run)+"/")
        for val in range(2,10,1):
            generate(n_values=val, n_actions=2, n_agents=30, pvs_prip=0.3, va_p=0.3, p_group_factor=0.5,
                    mu_p=0.85, va_mu=0.85, pvs_filename="vary_values_"+str(val), prip_filename="vary_values_"+str(val), pvs_output_dir=output_dir+"vary_vals/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_vals/PriP/"+str(run)+"/")
        for act in range(1, 10, 1):
            generate(n_values=4, n_actions=act, n_agents=30, pvs_prip=0.3, va_p=0.3, p_group_factor=0.5,
                     mu_p=0.85, va_mu=0.85, pvs_filename="vary_actions_" + str(act), prip_filename="vary_actions_" + str(act), pvs_output_dir=output_dir+"vary_acts/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_acts/PriP/"+str(run)+"/")

        ## MAJ/MIN SPLIT
        grp_facts = np.linspace(0.5, 1, 50)
        for grp_fact in grp_facts:
            generate(n_values=4, n_actions=2, n_agents=30, pvs_prip=0.3, va_p=0.3, p_group_factor=grp_fact,
                     mu_p=0.85, va_mu=0.85, pvs_filename="vary_grp_fact_"+str(grp_fact), prip_filename="vary_grp_fact_"+str(grp_fact), pvs_output_dir=output_dir+"vary_grp_fact/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_grp_fact/PriP/"+str(run)+"/")

        ## EXTREME PVSs
        mupvamu = np.linspace(0.5, 1, 50)
        for mup_vamu in mupvamu:
            generate(n_values=4, n_actions=2, n_agents=30, pvs_prip=0.3, va_p=0.3, p_group_factor=0.5,
                     mu_p=mup_vamu, va_mu=mup_vamu, pvs_filename="vary_mup_vamu_"+str(mup_vamu), prip_filename="vary_mup_vamu_"+str(mup_vamu), pvs_output_dir=output_dir+"vary_mup_vamu/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_mup_vamu/PriP/"+str(run)+"/")

        ## PriP NOISE
        pvs_prips = np.linspace(0, 1, 50)
        for pvs_prip in pvs_prips:
            generate(n_values=4, n_actions=2, n_agents=30, pvs_prip=pvs_prip, va_p=0.3, p_group_factor=0.5,
                     mu_p=0.85, va_mu=0.85, pvs_filename="vary_pvs_prip_"+str(pvs_prip), prip_filename="vary_pvs_prip_"+str(pvs_prip), pvs_output_dir=output_dir+"vary_pvs_prip/PVS/"+str(run)+"/", prip_output_dir=output_dir+"vary_pvs_prip/PriP/"+str(run)+"/")

