import argparse as ap
import numpy as np
import csv
import copy
from datetime import datetime as dt
from lp_regression.matrices import FormalisationObjects, FormalisationMatrix
from lp_regression.solve import L1, L2, Linf, Lp, mLp, transition_point, aggregate, aggregate_all_p, aggregate_prefs_only, aggregate_slm, aggregate_inf, aggregate_one
from files import limit_output, save_metadata, output_single, output_file
import pandas as pd
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
    parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    parser.add_argument('-w', type=int, default=0, help='Weights')
    ## FILE ARGS
    parser.add_argument('-f', type=str, default="value_systems/PVS_abstracted.csv", help='CSV file with personal values value_systems')
    parser.add_argument('-pf', type=str, default="value_systems/PriP_3q.csv", help='CSV file with principle value_systems')
    parser.add_argument('-slmf', type=str, default="input_data/sml_principles/placeolder_sml.csv", help='CSV file with principles for Salas-Molina method SML')
    ## COMPUTE ARGS
    parser.add_argument('-hcva', default=False, help='Compute HCVA', action='store_true')
    parser.add_argument('-hcva2', default=False, action='store_true', help='Compute HCVA++')
    parser.add_argument('-slm', default=False, action='store_true', help="Generate consensus using the method described by Salas-Molina et al.")
    parser.add_argument('-t', default=False, help='compute the threshold p, the transition point', action='store_true')
    parser.add_argument('-b', default=False, help='Compute the baseline aggregations (p=1. p=\infty)', action='store_true')
    parser.add_argument('-range', default=False, action='store_true', help='Aggregate everything')
    parser.add_argument('-range_step', type=float, default=0.1, help='Step size for args.range (aggregate everything)')
    # Initialise args and params
    args = parser.parse_args()
    n = args.n
    m = args.m
    # P_list, J_list, country_dict are the matrices created from the personal values
    P_list, J_list, w, country_dict = FormalisationObjects(filename=args.f, delimiter=',', weights=args.w)
    df = pd.read_csv(args.f)
    values_list = list([col for col in df.columns if 'P__' in col])
    actions_list = list([col for col in df.columns if 'VA__' in col])

    ## AGGREGATIONS/COMPUTE
    if args.t:
        """ Compute the transition point, and find an aggregation with that transition point P """
        now = dt.now().isoformat()
        # 1. Compute transition point
        p_list, dist_p_list, dist_inf_list, diff_list, t_point = transition_point(P_list, J_list, w, args.e)
        filename_limits = now + "_limits.csv"
        limit_output(
            p_list,
            dist_p_list,
            dist_inf_list,
            diff_list,
            filename_limits)
        # 2. Aggregate and store to a file.
        filename = str("T_"+now+".csv")
        filename_metadata = str("T_METADATA_"+now+".csv")
        p, u_pref, cons_pref = aggregate(P_list, J_list, w, t_point, True)
        _, u_act, cons_act = aggregate(P_list, J_list, w, t_point, False)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, t_point, _, _)
    elif args.hcva2:
        """ Compute HCVA++ (mean/JAIR) """
        print("Computing HCVA++")
        # 1. Find the consensus principle $p$
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
        _, _, _, _, transition_p =  transition_point(P_list, J_list, w, args.e)
        
        # 1.3 Given the transition point (best_p), find the consensus p by finding the
        # p the relative distance away from the transition point.
        consensus_p = pow(transition_p, (2*consensus_preference))
        print("Consensus p is: ", consensus_p)
        # 2. Aggregate all the preference values and action judgements submitted by agents
        # using the average rule as described in the paper. Do this twice, once for vals, other for action judgements
        now = dt.now().isoformat()
        filename = str("HCVApp_"+now+".csv")
        filename_metadata = str("HCVApp_METADATA_"+now+".csv")
        p, u_pref, cons_pref = aggregate(P_list, J_list, w, consensus_p, True)
        _, u_act, cons_act = aggregate(P_list, J_list, w, consensus_p, False)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, transition_p, consensus_p, consensus_preference)
    elif args.slm:
        """ Compute aggregation with Salas-Molina et al. baseline (Many P's) """
        print("Computing SLM")
        # 1. Read in the principles file. Each column contains a set of principles to use.
        file_path = args.slmf
        principles = pd.read_csv(file_path)
        # 2. For each list of ps in principles, aggregate and save
        for series_name, series in principles.items():
            now = dt.now().isoformat()
            filename= str("SLM_" + series_name + "_" + now + ".csv")
            filename_metadata = str("SLM_METADATA_" + series_name + "_" + now + ".csv")
            p, u_pref, cons_pref = aggregate_slm(P_list, J_list, w, series, True)
            _, u_act, cons_act = aggregate_slm(P_list, J_list, w, series, False)
            output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
            save_metadata(filename_metadata, args, _, series, _)
    elif args.hcva:
        """ Compute HCVA (closest P/VALE) """
        print("Computing HCVA")
        # 1. Formalise the principle preferences as matrices
        Pri_P_list, Pri_J_list, Pri_w, Pri_Country_dict = FormalisationObjects(
            filename=args.pf, delimiter=',', weights=args.w)
        # 2. Aggregate over all principle preferences
        p_list, _, cons_list, _, _, cons_1, cons_l = aggregate_prefs_only(Pri_P_list, Pri_J_list, Pri_w)
        # 3. Find a cutoff point given $\epsilon$
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
        print("Nearest P to mean con_vals is: ", con_p)
        # 6. Aggregate using that value of p
        # 7. Aggregate all preference values and action judgements submitted by agents
        # using the average rule as described in the paper. Do this twice, once for vals, other for action judgements
        now = dt.now().isoformat()
        filename = str("HCVA_" + now + ".csv")
        filename_metadata = str("HCVA_METADATA_" + now + ".csv")
        p, u_pref, cons_pref = aggregate(P_list, J_list, w, con_p, True)
        _, u_act, cons_act = aggregate(P_list, J_list, w, con_p, False)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, _, con_p, _)
    elif args.b:
        """ Compute aggregation for all other baselines (Util, Egal) """
        baseline_ps = [1, np.inf()]
        now = dt.now().isoformat()
        for p in baseline_ps:
            # Generate filenames
            filename = str(p + "_PERSONALS_" + now + ".csv")
            filename_metadata = str(p + "_METADATA_" + now + ".csv")
            # Aggregate and store
            if p == np.inf:
                _, u_pref, cons_pref = aggregate_inf(P_list, J_list, w, p, True)
                _, u_act, cons_act = aggregate_inf(P_list, J_list, w, p, False)
            elif p == 1:
                _, u_pref, cons_pref = aggregate_one(P_list, J_list, w, p, True)
                _, u_act, cons_act = aggregate_one(P_list, J_list, w, p, False)
            else:
                # Some other singular p
                _, u_pref, cons_pref = aggregate(P_list, J_list, w, p, True)
                _, u_act, cons_act = aggregate(P_list, J_list, w, p, False)
            output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
            save_metadata(filename_metadata, args, _, p, _)
    elif args.range:
        """Aggregate everything between 1-10 + infty with a step size defined as -step_size"""
        p_list, u_list, cons_list, dist1p_list, distpl_list, cons_1, cons_l = aggregate_all_p(P_list, J_list, w, args.range_step, True)
        # Save to a file
        now = dt.now().isoformat()
        filename = str("COMPLETE_PERSONALS_" + now + ".csv")
        filename_metadata = str("COMPLETE_METADATA_" + now + ".csv")
        output_file(p_list, u_list, cons_list, dist1p_list, distpl_list, cons_1, cons_l, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, _, p_list, _)
