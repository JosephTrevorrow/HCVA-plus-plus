import argparse as ap
import copy
import csv

import numpy as np
import os
from lp_regression.matrices import FormalisationObjects, FormalisationMatrix, principle_formalisation_objs
from files import output_file, limit_output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

# Julia for local machine
#import juliapkg
#juliapkg.require_julia("=1.10.3")
#juliapkg.resolve()
#from juliacall import Main as jl

## Julia for HPC
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import PyCall

## L_P REGRESSION FUNCTIONS HERE
# Note that functions L1, L2, Linf, IRLS,  and Lp are taken from the paper "Aggregating Value Systems for Decision Support" https://www.sciencedirect.com/science/article/pii/S0950705124000881
def L1(A, b):
    """
    This function runs the L1 norm on values and returns consensus.
    Note that this is the fully utilitarian case P=1
    OUTPUT:
    cons - the consensus matrix in the same format as the P or J matrix inputted
    r - The value of the solved function ||Ax - b||
    u - The distance between the value of the solved function ||Ax - b|| and 1
    """
    # create variables
    l = A.shape[1]
    t = cp.Variable(len(b), integer=False)
    x = cp.Variable(l, integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t]
    constraint2 = [A @ x - b <= t]
    constraints = constraint1 + constraint2
    cost = cp.sum(t)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # optimise model
    prob.solve(solver='ECOS', verbose=False, solver_verbose=False)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    #print("obj value:", obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L2(A, b):
    """
    This function runs the L2 norm on values and returns consensus
    P=2
    """
    cons, res, rank, a = np.linalg.lstsq(A, b, rcond=None)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r)

def Linf(A, b):
    """
    This function runs the Linf norm on values and returns consensus
    OUTPUT:
    cons - the consensus matrix in the same format as the P or J matrix inputted
    r - The value of the solved function ||Ax - b||
    u - The distance between the value of the solved function ||Ax - b|| and np.inf
    """
    # create variables
    l = A.shape[1]
    t = cp.Variable(1, integer=False)
    x = cp.Variable(l, integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t * np.ones_like(b)]
    constraint2 = [A @ x - b <= t * np.ones_like(b)]
    constraints = constraint1 + constraint2
    prob = cp.Problem(cp.Minimize(t), constraints)
    # optimise model
    prob.solve(solver='ECOS', verbose=False, solver_verbose=False)
    # prob.solve(solver='GLPK', verbose=True)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    #print("obj value: ", obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)

def IRLS(A, b, p, max_iter=int(1e6), e=1e-3, d=1e-4):
    """
    This function runs the IRLS method for finding consensus for any P >= 3
    using a python implementation
        OUTPUT:
    cons - the consensus matrix in the same format as the P or J matrix inputted
    r - The value of the solved function ||Ax - b||
    u - The distance between the value of the solved function ||Ax - b|| and p
    """
    # l = A.shape[1]
    n = A.shape[0]
    D = np.repeat(d, n)
    W = np.diag(np.repeat(1, n))
    x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b  # initial LS solution
    for i in range(max_iter):
        W_ = np.diag(np.power(np.maximum(np.abs(b - A @ x), D), p - 2))
        # reweighted LS solution
        x_ = np.linalg.inv(A.T @ W_ @ A) @ A.T @ W_ @ b
        e_ = sum(abs(x - x_))
        if e_ < e:
            break
        else:
            W = W_
            x = x_
    r = np.abs(A @ x - b)
    return x, r, np.linalg.norm(r, p)

def Lp(A, b, p):
    """OUTPUT:
    cons - the consensus matrix in the same format as the P or J matrix inputted
    r - The value of the solved function ||Ax - b||
    u - The distance between the value of the solved function ||Ax - b|| and p"""
    # l = A.shape[1]
    if p >= 2 :  # pIRLS implementation (NIPS 2019) (always use this for continuity)

        ## Local machine
        #jl.include(os.path.dirname(
        #        os.path.realpath(__file__)) +
        #    '/IRLS-pNorm.jl')

        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        # jl.pNorm has the following parameters:
        # ϵ : accuracy we want to achieve
        # A,b : the objective we are minimizing is ||Ax-b||_p^p
        # p : the norm we want to minimize
        # C,d : The linear constraints are Cx = d
        # x : Initial solution
        # lb : lower bound on the optimum
        # function pNorm(ϵ,A,b,p,C,d, x, lb)
        cons, it = Main.MyActionModule.pNorm(epsilon, A, b.reshape(-1, 1),
                              p, C, d.reshape(-1, 1))
        # So the cons we return is the same as
        # cons, it = IRLS.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        Main.MyActionModule.collector()
        return cons, r, np.linalg.norm(r, p)
    else:  # vanilla IRLS implementation
        return IRLS(A, b, p)

def Lp_norm(A,b,p, v):
    """ This function is taken from https://github.com/filippobistaffa/social-choice-pnorm. It should be identical in output to Lp, but to be certain that the SLM
    baseline is computing as intended, we use their solver here. (The main difference is that SLM uses CPLEX, but other methods use ECOS)"""
    x = cp.Variable(v)
    cost = cp.pnorm(A @ x - b, p)
    prob = cp.Problem(cp.Minimize(cost))
    print("Lp_norm solving")
    prob.solve(solver='GUROBI', verbose=True)
    return prob.value

def mLp(A, b, ps, λs, weight=True):
    """
    This function is used by the -slm arg to run the mLp method for finding consensus using multiple p values.
    This function is taken from the following repo: https://github.com/filippobistaffa/social-choice-pnorm    """
    v = A.shape[1]
    wps = [λ / Lp_norm(A, b, p, v) if weight else λ for λ, p in zip(λs, ps)]
    x = cp.Variable(v)
    print("X", v)
    constraints = [x == 16]
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost), constraints=constraints)
    print("mLp solving")
    prob.solve(solver="GUROBI", verbose=True, warm_start=True)
    #res = np.abs(A @ x.value - b)
    #psi = np.var([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, None, prob.value / sum(wps), None

#### RUNNER FUNCTIONS HERE ######

def find_transition_and_aggregate(P_list, J_list, w, output_dir, filename_limits, e, args):
    """ Compute the transition point, and find an aggregation with that transition point P """
    # 1. Compute transition point
    p_list, dist_p_list, dist_inf_list, diff_list, t_point = transition_point(P_list, J_list, w, e)
    #limit_output(
    #    p_list,
    #    dist_p_list,
    #    dist_inf_list,
    #    diff_list,
    #    output_dir,
    #    filename_limits)
    # 2. Aggregate and store to a file.
    p, u_pref, cons_pref = aggregate(P_list, J_list, w, t_point, True)
    _, u_act, cons_act = aggregate(P_list, J_list, w, t_point, False)
    return p, u_pref, cons_pref, u_act, cons_act, t_point

def find_hcva_and_aggregate(P_list, J_list, w, prip_df, args):
    # 1. Formalise the principle preferences as matrices
    Pri_P_list, _, Pri_w, Pri_Country_dict = principle_formalisation_objs(
        prip_df, weights=args.w)
    # 2. Aggregate over all principle preferences
    p_list, _, cons_list, _, _, cons_1, cons_l = aggregate_prefs_only(Pri_P_list, [], Pri_w)
    # 3. Find a cutoff point given $\epsilon$
    cut_point = 10
    incr = 0.1
    j = 0
    # This version of epsilon is the same as used in original paper, defined arbitrarily in that case.
    epsilon = args.e
    for i in np.arange(1, 10, incr):
        cons = cons_list[j]
        if __debug__:
            print("cons: ", cons)
            print("cons_1: ", cons_1)
            print("cons_l: ", cons_l)
        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        j += 1
        if __debug__:
            print("dist_1p: ", dist_1p, " dist_pl: ", dist_pl, " i: ", i, "")
            print("abs(dist_1p - dist_pl): ", abs(dist_1p - dist_pl), " epsilon: ", epsilon)
        if abs(dist_1p - dist_pl) < epsilon:
            cut_point = i
            if __debug__:
                print('Not improving anymore at cut_point = ', cut_point, '. Stopping...')
            break
    # 4. Cut the list of consensuses using the cut_point, find mean
    cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
    con_vals = [sum(i[0] for i in cut_list) / len(cut_list), sum(i[1] for i in cut_list) / len(cut_list)]

    # 5. Find the value of p from the mean of these consensuses
    con_p = 1.0
    best_dist = 999
    for j in range(len(cut_list)):
        dist = [abs(cut_list[j][0] - con_vals[0]), abs(cut_list[j][1] - con_vals[1])]
        dist = sum(dist)
        if dist < best_dist:
            best_dist = dist
            # to convert from ordinal list num to corresponding p
            con_p = (j / 10) + 1
    #print("Nearest P to mean con_vals is: ", con_p)
    p, u_pref, cons_pref = aggregate(P_list, J_list, w, con_p, True)
    _, u_act, cons_act = aggregate(P_list, J_list, w, con_p, False)
    return p, u_pref, u_act, cons_pref, cons_act, con_p

def find_hcva_pp_and_aggregate(P_list, J_list, w, prip_df, transition_p, args):
    """ Compute the HCVA++ consensus principle, and find an aggregation with that consensus principle P """
    # 1. Find the consensus principle $p$
    # 1.1 Find the consensus principle preference
    principle_preferences = prip_df["Egalitarian"].astype("float").values.tolist()
    print("Principle preferences: ", principle_preferences)

    consensus_preference = sum(principle_preferences) / len(principle_preferences)
    consensus_preference = round(consensus_preference, 2)
    print("HCVA++ Consensus preference is: ", consensus_preference)
    # 1.2 Aggregate personal values/action judgements to find the transition point - Not needed if t_point provided
    if transition_p is None:
        _, _, _, _, transition_p = transition_point(P_list, J_list, w, args.e)
    # 1.3 Given the transition point (best_p), find the consensus p by finding the
    # p the relative distance away from the transition point.
    consensus_p = pow(transition_p, (2 * consensus_preference))
    # Round to 2 d.p. for fairness
    consensus_p = round(consensus_p, 2)
    print("Consensus p is: ", consensus_p)
    # 2. Aggregate all the preference values and action judgements submitted by agents
    # using the average rule as described in the paper. Do this twice, once for vals, other for action judgements
    p, u_pref, cons_pref = aggregate(P_list, J_list, w, consensus_p, True)
    _, u_act, cons_act = aggregate(P_list, J_list, w, consensus_p, False)
    return p, u_pref, cons_pref, u_act, cons_act, consensus_p, transition_p, consensus_preference

def find_slm_and_aggregate(P_list, J_list, w, prip_df, transition_p, args):
    """ Compute aggregation with Salas-Molina et al. baseline (Many P's) """
    principle_preferences = prip_df["Egalitarian"].astype("float").values.tolist()
    # Convert the principles (which are preferences) into numbers (need to first find transition point
    if transition_p is None:
        _, _, _, _, transition_p = transition_point(P_list, J_list, w, args.e)
    converted_principles = []
    for principle in principle_preferences:
        # Find p by finding the p the relative distance away from the transition point.
        converted_p = pow(transition_p, (2 * principle))
        # Round to 2 d.p. for fairness
        converted_p = round(converted_p, 2)
        converted_p = max(1, converted_p)
        converted_principles.append(float(converted_p))
    p, _, cons_pref = aggregate_slm(P_list, J_list, w, converted_principles, True)
    _, _, cons_act = aggregate_slm(P_list, J_list, w, converted_principles, False)
    return p, _, cons_pref, _, cons_act, converted_principles

def transition_point(P_list, J_list, w, e):
    """
    Find the transition point given personal values
    """
    # Cons values are a flattened consensus matrix for either preferences or actions.
    # Join the two consensus lists of preferences and action judgements
    #   to get a single consensus list for p=1 and p=\infty
    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1_pref, _, _ = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, 1, False)
    cons_1_act, _, _ = L1(A, b)
    # Cut the actions in half, as it produces two sets of consensuses -> J_p and J_n
    cons_1_act = cons_1_act[:len(cons_1_act)//2]
    cons_1 = np.concatenate((cons_1_pref, cons_1_act))

    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l_pref, _, _ = Linf(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, False)
    cons_l_act, _, _ = Linf(A, b)
    cons_l_act = cons_l_act[:len(cons_l_act)//2]
    cons_l = np.concatenate((cons_l_pref, cons_l_act))

    diff = np.inf
    incr = 0.01
    p_list = []
    dist_p_list = []
    dist_inf_list = []
    diff_list = []
    # Check all values until 10
    p = 10
    best_p = 0 # base val
    for i in np.arange(1 + incr, p, incr):
        A, b = FormalisationMatrix(P_list, J_list, w, i, True)
        cons_pref, _, u_pref = Lp(A, b, i)
        A, b = FormalisationMatrix(P_list, J_list, w, i, False)
        cons_act, _, u_act = Lp(A, b, i)
        cons_act = cons_act[:len(cons_act) // 2]

        cons = np.concatenate((cons_pref, cons_act))
        #print('p: {:.2f}, cons: '.format(i), cons)
        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        # Go to at least 3. SLM note that p >= 3 is not significantly different to p = \infty
        if i > 3.0:
            #abs(dist_1p - dist_pl) < e and
            best_p = i
            if __debug__:
                print('Not improving anymore, stopping!')
            # Stop this for performance.
            return p_list, dist_p_list, dist_inf_list, diff_list, best_p
        else:
            #if __debug__:
                #print('p = {:.2f}'.format(i))
                #print('Distance L1<-->L{:.2f} = {:.4f}'.format(i, dist_1p))
                #print(
                #    'Distance L{:.2f}<-->L{:.2f} = {:.4f}'.format(i, p, dist_pl))
                #print(
                #    'Difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(
                #        i, i, p, abs(
                #            dist_1p - dist_pl)))
                #print(
                #    'Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, best_p, diff))
            if abs(dist_1p - dist_pl) < diff:
                diff = abs(dist_1p - dist_pl)
                best_p = i
            p_list.append(i)
            dist_p_list.append(dist_1p)
            dist_inf_list.append(dist_pl)
            diff_list.append(abs(dist_1p - dist_pl))  
        #print('Transition point: {:.2f}'.format(best_p))
    return p_list, dist_p_list, dist_inf_list, diff_list, best_p

def aggregate(P_list, J_list, w, p, v):
    """Compute one aggregation using the P specified"""
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = Lp(A, b, p)
    if __debug__:
        print('Aggregate: p: {:.2f}, cons: '.format(p), cons)
    if not v:
        cons = cons[:len(cons) // 2]
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    #print('p: {:.2f}, cons: '.format(p), cons)
    return p, u, cons

def aggregate_all_p(P_list, J_list, w, incr):
    """This function aggregates over all P between 1-10, given a step size"""
    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1_pref, r_1_pref, u_1_pref = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, 1, False)
    cons_1_act, r_1_act, u_1_act = L1(A, b)
    cons_1_act = cons_1_act[:len(cons_1_act) // 2]
    cons_1 = np.concatenate((cons_1_pref, cons_1_act))
    u = np.array([u_1_pref, u_1_act])


    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l_pref, _, _ = Linf(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, False)
    cons_l_act, _, _ = Linf(A, b)
    cons_l_act = cons_l_act[:len(cons_l_act) // 2]
    cons_l = np.concatenate((cons_l_pref, cons_l_act))

    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    # print('{:.2f} \t \t {:.4f}'.format(p, ua))
    p_list = [1.0]
    u_list = [u]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    while p < 10:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, True)
        cons_pref, _, u_pref = Lp(A, b, p)
        A, b = FormalisationMatrix(P_list, J_list, w, p, False)
        cons_act, _, u_act = Lp(A, b, p)
        cons_act = cons_act[:len(cons_act) // 2]
        cons = np.concatenate((cons_pref, cons_act))
        u = np.array([u_pref, u_act])

        p_list.append(p)
        u_list.append(u)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        # print('{:.2f} \t \t {:.4f}'.format(p, ub))
    return p_list, u_list, cons_list, dist_1p_list, dist_pl_list, cons_1, cons_l

def aggregate_prefs_only(P_list, J_list, w):
    """This function is used by the HCVA to aggregate over all principle preferences in main.py"""
    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1_pref, _, u_1_pref = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l_pref, _, _ = Linf(A, b)

    dist_1p = np.linalg.norm(cons_1_pref - cons_1_pref, 1)
    dist_pl = np.linalg.norm(cons_l_pref - cons_1_pref, np.inf)
    p = 1
    # print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [u_1_pref]
    cons_list = [cons_1_pref]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    while p < 10:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, True)
        cons_pref, _, u_pref = Lp(A, b, p)
        p_list.append(p)
        u_list.append(u_pref)
        cons_list.append(cons_pref)
        dist_1p = np.linalg.norm(cons_1_pref - cons_pref, p)
        dist_pl = np.linalg.norm(cons_l_pref - cons_pref, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        # print('{:.2f} \t \t {:.4f}'.format(p, ub))
    return p_list, u_list, cons_list, dist_1p_list, dist_pl_list, cons_1_pref, cons_l_pref

def aggregate_slm(P_list, J_list, w, list_of_ps, v):
    p_list = []
    u_list = []
    cons_list = []
    dist_1p_list = []
    dist_pl_list = []
    # Form a matrix.
    ps = np.atleast_1d(list_of_ps)
    ps = np.where(ps == -1, np.inf, ps)
    λs = np.ones_like(ps)
    nλs = min(len(λs), len([]))
    λs[:nλs] = [][:nλs]
    A, b = FormalisationMatrix(P_list, J_list, w, 1, v)
    # w will always have weights equal to 1, shape needs to be equal. We do not use weights in the paper for simplicity.
    w = np.repeat(w, A.shape[1])
    if __debug__:
        print("A dtype:", np.asarray(A).dtype)
        print("A shape:", np.asarray(A).shape)
        print("b dtype:", np.asarray(b).dtype)
        print("b shape:", np.asarray(b).shape)
        print("A finite:", np.isfinite(np.asarray(A, dtype=float)).all())
        print("b finite:", np.isfinite(np.asarray(b, dtype=float)).all())
        print("A min/max:", np.min(np.asarray(A, dtype=float)), np.max(np.asarray(A, dtype=float)))
        print("b min/max:", np.min(np.asarray(b, dtype=float)), np.max(np.asarray(b, dtype=float)))
        print("Type of min:", type(np.min(np.asarray(A, dtype=float)).item()))
    # Aggregate over all principles together using the matrix
    cons, _, _, _ = mLp(A, b, ps, λs, False)
    return list_of_ps, _, cons

def aggregate_inf(P_list, J_list, w, p, v):
    # Compute one aggregation using the P specified
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = Linf(A, b)
    if __debug__:
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))
        print('p: {:.2f}, cons: '.format(p), cons)
    return p, u, cons

def aggregate_one(P_list, J_list, w, p, v):
    # Compute one aggregation using the P specified
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = L1(A, b)
    if __debug__:
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))
        print('p: {:.2f}, cons: '.format(p), cons)
    return p, u, cons