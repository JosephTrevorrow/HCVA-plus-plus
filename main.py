import argparse as ap
import numpy as np
import os
import csv
import copy
from datetime import datetime as dt

from fontTools.misc.bezierTools import epsilon

from lp_regression.matrices import FormalisationObjects, FormalisationMatrix
from lp_regression.solve import L1, L2, Linf, Lp, mLp, transition_point, aggregate, aggregate_all_p
from files import output_file, limit_output, save_metadata
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import juliapkg
juliapkg.require_julia("=1.10.3")
juliapkg.resolve()
from juliacall import Main as jl

np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)


if __name__ == '__main__':
    parser = ap.ArgumentParser()

    ## PARAMETER ARGS
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=2, help='m')
    parser.add_argument('-p', type=float, default=10, help='p')
    parser.add_argument('-e', type=float, default=1e-4, help='e')
    parser.add_argument(
        '-w',
        type=int,
        default=0,
        help='Weights')

    ## FILE ARGS
    parser.add_argument(
        '-f',
        type=str,
        default="22-01-2025-agent-value_systems.csv",
        help='CSV file with personal values value_systems')
    parser.add_argument(
        '-pf',
        type=str,
        default="input_data/principles/placeholder_principles.csv",
        help='CSV file with principle value_systems')
    parser.add_argument(
        '-smlf',
        type=str,
        default="input_data/sml_principles/placeolder_sml.csv",
        help='CSV file with principles for Salas-Molina method SML'
    )    
    parser.add_argument(
        '-o', 
        type=str, 
        default="consensus_args_o.csv",
        help='write consensus arguments to file')
    parser.add_argument(
        '-g',
        type=str,
        default='none',
        help='store results in csv')
    
    ## COMPUTE ARGS
    parser.add_argument(
        '-v',
        default=False,
        help='computes the preference aggregation with p given in arg -p',
        action='store_true')    
    parser.add_argument(
        '-hcva',
        default=False,
        help='Compute HCVA',
        action='store_true'
        )    
    parser.add_argument(
        '-hcva2',
        default=True,
        action='store_true',
        help='Compute HCVA++'
    )
    parser.add_argument(
        '-sml',
        default=False,
        action='store_true',
        help="Generate consensus using the method described by Salas-Molina et al."
    )
    parser.add_argument(
        '-t',
        default=False,
        help='compute the threshold p, the transition point',
        action='store_true'
    )

    # Initialise args and params
    args = parser.parse_args()
    n = args.n
    m = args.m

    # P_list, J_list, country_dict are the matricies created from the personal values
    P_list, J_list, w, country_dict = FormalisationObjects(
        filename=args.f, delimiter=',', weights=args.w)
        
    ## AGGREGATIONS/COMPUTE
    if args.t:
        """
        Compute the transition point and save to file
        """
        p_list, dist_p_list, dist_inf_list, diff_list, _ = transition_point(P_list, J_list, w, args.v, args.e, p)
        limit_output(
            p_list,
            dist_p_list,
            dist_inf_list,
            diff_list,
            "limits.csv"
        )
    elif args.hcva2:
        """
        Compute HCVA (mean/JAIR)
        """
        print("Computing HCVA++")
        # 1. Find the consensus priciple $p$
        # 1.1 Find the consensus principle preference
        principle_preferences = []
        with open(args.pf) as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # get rid of the header row
            for row in reader:
                temp_preference = float(row[1])
                principle_preferences.append(copy.copy(temp_preference))
        consensus_preference = sum(principle_preferences) / len(principle_preferences)
        print("Consensus preference is: ", consensus_preference)
        consensus_preference = round(consensus_preference,2)
        # 1.2 Aggregate personal values/action judgements to find the transition point
        _, _, _, _, transition_p =  transition_point(P_list, J_list, w, args.v, args.e)
        
        # 1.3 Given the transition point (best_p), find the consensus p by finding the
        # p the relative distance away from the transition point.
        consensus_p = pow(transition_p, (2*consensus_preference))

        # 1.4 Aggregate all of the preference values, and action judgements submitted by agents
        # using the average rule as described in paper. Do this twice, once for vals, other for action judgements
        now = dt.now().isoformat()
        filename_personal_vals = str("HCVA_PERSONALS_"+now+".csv")
        filename_action_judgements = str("HCVA_ACTIONS_"+now+".csv")
        filename_metadata = str("HCVA_METADATA_"+now+".csv")

        aggregate(P_list, J_list, w, consensus_p, True, filename_personal_vals)
        aggregate(P_list, J_list, w, consensus_p, False, filename_action_judgements)

        save_metadata(filename_metadata, args, transition_p, consensus_p, consensus_preference)
    elif args.sml:
        """
        Compute aggregation with Salas-Molina et al. baseline (Many P's)
        """
        print("Computing SML")

        # 1. Read in the principles file. Each column contains a set of principles to use.
        file_path = args.smlf
        principles = pd.read_csv(file_path)

        consensus_df = pd.DataFrame()
        for series_name, series in principles.items():
            # 2. For each set of principle preferences, form a matrix.
            p = series
            ps = np.atleast_1d(p)
            ps = np.where(ps == -1, np.inf, ps)
            λs = np.ones_like(ps)
            nλs = min(len(λs), len([]))
            λs[:nλs] = [][:nλs]

            P_list, J_list, w, country_dict = FormalisationObjects(
                filename=args.f, delimiter=',', weights=args.w)
            A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
            # w will always have weights equal to 1, shape needs to be equal. We do not use weights in the paper for simplicity.
            w = np.repeat(w, A.shape[1])

            # 3. Aggregate over all principles together using the matrix
            cons, res, u, psi = mLp(A, b, ps, λs, False)

            P_list, J_list, w, country_dict = FormalisationObjects(
                filename=args.f, delimiter=',', weights=args.w)
            A_1, b_1 = FormalisationMatrix(P_list, J_list, w, 1, not (args.v))
            # w has weights equal to 1, shape needs to be equal
            w = np.repeat(w, A_1.shape[1])

            now = dt.now().isoformat()
            filename_personal_vals = str("HCVA_PERSONALS_" + now + ".csv")
            filename_action_judgements = str("HCVA_ACTIONS_" + now + ".csv")
            filename_metadata = str("HCVA_METADATA_" + now + ".csv")

            cons_1, res, u, psi = mLp(A_1, b_1, ps, λs, False, )

            # Note: mLp returns: x.value, res, prob.value / sum(wps), psi
            """ Artefact from HCVA VALE work, will check and remove.
            # NOTE: COLUMN NAMES ARE ONLY ACCURATE WHEN ARGS.V = FALSE
            consensus_df = consensus_df._append({
                'series_name': series_name,
                'Rel_div_p': cons[0],
                'Nonrel_div_p': cons[1],
                'Rel_div_n': cons[2],
                'Nonrel_div_n': cons[3],
                'Rel-Rel': cons_1[0],
                'Rel-Nonrel': cons_1[1],
                'Nonrel-Rel': cons_1[2],
                'Nonrel-Nonrel': cons_1[3]
            }, ignore_index=True)
            print("len of cons: ", len(cons))
            print("len of cons_1: ", len(cons_1))
            """
            # 4. Save to a file




            save_metadata(filename_metadata, args, _, con_p, _)

    elif args.hcva:
        """
        Compute HCVA (closest P/VALE)
        """
        print("Computing HCVA")
        #  1.1 Formalise the principle preferences as matrices
        Pri_P_list, Pri_J_list, Pri_w, Pri_Country_dict = FormalisationObjects(
            filename=args.pf, delimiter=',', weights=args.w)

        # 1.2 Aggregate over all principle preferences
        p_list, _, cons_list, _, _, cons_1, cons_l = aggregate_all_p(Pri_P_list, Pri_J_list, Pri_w)

        # 1.3 Find a cutoff point given $\epsilon$
        cut_point = 10
        incr = 0.1
        j = 0
        epsilon = 0.05
        for i in np.arange(1 + incr, 10, incr):
            cons = cons_list[j]
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            j += 1
            if abs(dist_1p - dist_pl) < epsilon:
                cut_point = i
                print('Not improving anymore at cut_point = ', cut_point, '. Stopping...')
                break
        # 1.4 Cut the list of consensuses using the cut_point, find mean
        cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
        con_vals = [sum(i[0] for i in cut_list) / len(cut_list), sum(i[1] for i in cut_list) / len(cut_list)]

        # 1.5 Find the value of p from the mean of these consensuses
        con_p = 1.0
        best_dist = 999
        for j in range(len(cut_list)):
            dist = [abs(cut_list[j][0] - con_vals[0]), abs(cut_list[j][1] - con_vals[1])]
            dist = sum(dist)
            if dist < best_dist:
                best_dist = dist
                # to convert from ordinal list num to corresponding p
                con_p = (j / 10) + 1
        print("Nearest P to mean con_vals is: ", con_p)

        # 1.6 Aggregate using that value of p
        # 1.4 Aggregate all of the preference values, and action judgements submitted by agents
        # using the average rule as described in paper. Do this twice, once for vals, other for action judgements
        now = dt.now().isoformat()
        filename_personal_vals = str("HCVA_PERSONALS_" + now + ".csv")
        filename_action_judgements = str("HCVA_ACTIONS_" + now + ".csv")
        filename_metadata = str("HCVA_METADATA_" + now + ".csv")

        aggregate(P_list, J_list, w, con_p, True, filename_personal_vals)
        aggregate(P_list, J_list, w, con_p, False, filename_action_judgements)

        save_metadata(filename_metadata, args, _, con_p, _)
    elif args.v:
        """
        Compute aggregation with set P (Lera-Leri et al., Util, Egal, T etc.)
        P (principle) value must be given
        """
        print("Computing aggregation with set P: ", args.p)
        p = args.p

        