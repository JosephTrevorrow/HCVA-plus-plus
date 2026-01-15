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
    Note that this is the fully egalitarian case P=inf
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
    # l = A.shape[1]
    if p >= 2 :  # pIRLS implementation (NIPS 2019) (always use this for continuity)
        jl.include(os.path.dirname(
                os.path.realpath(__file__)) +
            '/IRLS-pNorm.jl')
        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        cons, it = jl.pNorm(epsilon, A, b.reshape(-1, 1),
                              p, C, d.reshape(-1, 1))
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
    prob.solve(solver="ECOS", verbose=True)
    res = np.abs(A @ x.value - b)
    psi = np.var([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps), psi

## RUNNER FUNCTIONS HERE
def transition_point(P_list, J_list, w, v, e, p):
    """
    Find the transition point given personal values
    """
    # Join the two consensus lists of preferences and action judgements
    #   to get a single consensus list for p=1 and p=\infty
    A, b = FormalisationMatrix(P_list, J_list, w, 1, v)
    cons_1, r_1, u_1 = L1(A, b)
    cons_1 = cons_1[1:3]
    A, b = FormalisationMatrix(P_list, J_list, w, 1, not(v))
    cons_1_1, r_1_1, u_1_1 = L1(A, b)
    cons_1_1 = cons_1_1[1:3]
    cons_1 = np.concatenate((cons_1, cons_1_1))
    
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, v)
    cons_l, r_l, u_l = Linf(A, b)
    cons_l = cons_l[1:3]
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, not(v))
    cons_l_1, r_l_1, u_l_1 = Linf(A, b)
    cons_l_1 = cons_l_1[1:3]
    cons_l = np.concatenate((cons_l, cons_l_1))

    diff = np.inf
    incr = 0.1

    p_list = []
    dist_p_list = []
    dist_inf_list = []
    diff_list = []
    for i in np.arange(1 + incr, p, incr):
        A, b = FormalisationMatrix(P_list, J_list, w, i, v)
        cons, r, u = Lp(A, b, i)
        cons = cons[1:3]
        A, b = FormalisationMatrix(P_list, J_list, w, i, not(v))
        cons_cons, r_cons, u_cons = Lp(A, b, i)
        cons_cons = cons_cons[1:3]
        cons = np.concatenate((cons, cons_cons))
        print('p: {:.2f}, cons: '.format(i), cons)

        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        if (abs(dist_1p - dist_pl) < e):
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
                'Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, diff))
            if abs(dist_1p - dist_pl) < diff:
                diff = abs(dist_1p - dist_pl)
                best_p = i
            p_list.append(i)
            dist_p_list.append(dist_1p)
            dist_inf_list.append(dist_pl)
            diff_list.append(abs(dist_1p - dist_pl))  
        print('Transition point: {:.2f}'.format(best_p))
    return p_list, dist_p_list, dist_inf_list, diff_list, best_p

def aggregate(P_list, J_list, w, p, v, filename):
    A, b = FormalisationMatrix(P_list, J_list, w, 1, v)
    cons_1, _, ua = Lp(A, b, 1)
    print('L1 =', cons_1)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, v)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    # Compute one aggregation using the P specified
    A, b = FormalisationMatrix(P_list, J_list, w, p, v)
    cons, _, ub = Lp(A, b, p)
    p_list.append(p)
    u_list.append(ub)
    cons_list.append(cons)
    dist_1p = np.linalg.norm(cons_1 - cons, p)
    dist_pl = np.linalg.norm(cons_l - cons, p)
    dist_1p_list.append(dist_1p)
    dist_pl_list.append(dist_pl)
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    print('p: {:.2f}, cons: '.format(p), cons)

    output_file(
        p_list,
        u_list,
        cons_list,
        dist_1p_list,
        dist_pl_list,
        v,
        filename)