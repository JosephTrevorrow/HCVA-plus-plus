import argparse as ap
import numpy as np
import os
from lp_regression.matrices import FormalisationObjects, FormalisationMatrix
from files import output_file, limit_output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import juliapkg
juliapkg.require_julia("=1.10.3")
juliapkg.resolve()
from juliacall import Main as jl

## L_P REGRESSION FUNCTIONS HERE
def L1(A, b):
    """
    This function runs the L1 norm on values and returns consensus.
    Note that this is the fully utilitarian case P=1
    OUTPUT:
    cons - the consensus matrix in the same format as the P or J matrix inputted
    r - The value of the solved function ||Ax - b||
    u - The distance between the value of the solved function ||Ax - b|| and 1
    """
    import cvxpy as cp
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
    # optimize model
    prob.solve(solver='ECOS', verbose=True)
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
    import cvxpy as cp
    # create variables
    l = A.shape[1]
    t = cp.Variable(1, integer=False)
    x = cp.Variable(l, integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t * np.ones_like(b)]
    constraint2 = [A @ x - b <= t * np.ones_like(b)]
    constraints = constraint1 + constraint2
    prob = cp.Problem(cp.Minimize(t), constraints)
    # optimize model
    prob.solve(solver='ECOS', verbose=False)
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
        jl.include(os.path.dirname(
                os.path.realpath(__file__)) +
            '/IRLS-pNorm.jl')
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
        cons, it = jl.pNorm(epsilon, A, b.reshape(-1, 1),
                              p, C, d.reshape(-1, 1))
        # So the cons we return is the same as
        # cons, it = IRLS.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        jl.collector()
        return cons, r, np.linalg.norm(r, p)
    else:  # vanilla IRLS implementation
        return IRLS(A, b, p)

def mLp(A, b, ps, λs, weight=True):
    """
    This function is used by the -slm arg to run the mLp method for finding consensus using multiple p values.
    This function is taken from the following repo: https://github.com/filippobistaffa/social-choice-pnorm
    """
    wps = [λ / Lp(A, b, p) if weight else λ for λ, p in zip(λs, ps)]
    v = A.shape[1]
    x = cp.Variable(v)
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost))
    for solver_name in ("CLARABEL", "SCS", "ECOS"):
        try:
            prob.solve(solver=solver_name, verbose=True)
            if x.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                print(f"mLp solved with {solver_name}.")
                break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"All solvers failed in mLp. Last error: {last_error}")

    res = np.abs(A @ x.value - b)
    psi = np.var([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps), psi

## RUNNER FUNCTIONS HERE
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
        print('p: {:.2f}, cons: '.format(i), cons)
        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        if abs(dist_1p - dist_pl) < e:
            best_p = i
            print('Not improving anymore, stopping!')
        else:
            print('p = {:.2f}'.format(i))
            print('Distance L1<-->L{:.2f} = {:.4f}'.format(i, dist_1p))
            print(
                'Distance L{:.2f}<-->L{:.2f} = {:.4f}'.format(i, p, dist_pl))
            print(
                'Difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(
                    i, i, p, abs(
                        dist_1p - dist_pl)))
            print(
                'Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, best_p, diff))
            if abs(dist_1p - dist_pl) < diff:
                diff = abs(dist_1p - dist_pl)
                best_p = i
            p_list.append(i)
            dist_p_list.append(dist_1p)
            dist_inf_list.append(dist_pl)
            diff_list.append(abs(dist_1p - dist_pl))  
        print('Transition point: {:.2f}'.format(best_p))
    return p_list, dist_p_list, dist_inf_list, diff_list, best_p

def aggregate(P_list, J_list, w, p, v):
    """Compute one aggregation using the P specified"""
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = Lp(A, b, p)
    print('Aggregate: p: {:.2f}, cons: '.format(p), cons)
    if not v:
        cons = cons[:len(cons) // 2]
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    print('p: {:.2f}, cons: '.format(p), cons)
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
    p = list_of_ps
    print("p: ", p)
    ps = np.atleast_1d(p)
    print("ps: ", ps)
    ps = np.where(ps == -1, np.inf, ps)
    λs = np.ones_like(ps)
    nλs = min(len(λs), len([]))
    λs[:nλs] = [][:nλs]
    A, b = FormalisationMatrix(P_list, J_list, w, 1, v)
    # w will always have weights equal to 1, shape needs to be equal. We do not use weights in the paper for simplicity.
    w = np.repeat(w, A.shape[1])
    # Aggregate over all principles together using the matrix
    print("A dtype:", np.asarray(A).dtype)
    print("A shape:", np.asarray(A).shape)
    print("b dtype:", np.asarray(b).dtype)
    print("b shape:", np.asarray(b).shape)
    print("A finite:", np.isfinite(np.asarray(A, dtype=float)).all())
    print("b finite:", np.isfinite(np.asarray(b, dtype=float)).all())
    print("A min/max:", np.min(np.asarray(A, dtype=float)), np.max(np.asarray(A, dtype=float)))
    print("b min/max:", np.min(np.asarray(b, dtype=float)), np.max(np.asarray(b, dtype=float)))
    cons, res, u, psi = mLp(A, b, ps, λs, False)
    return p, u, cons

def aggregate_inf(P_list, J_list, w, p, v):
    # Compute one aggregation using the P specified
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = Linf(A, b)
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    print('p: {:.2f}, cons: '.format(p), cons)
    return p, u, cons

def aggregate_one(P_list, J_list, w, p, v, filename):
    # Compute one aggregation using the P specified
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, u = L1(A, b)
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    print('p: {:.2f}, cons: '.format(p), cons)
    return p, u, cons