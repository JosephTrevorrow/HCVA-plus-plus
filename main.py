import argparse as ap
import numpy as np
import csv
import copy
from datetime import datetime as dt
from lp_regression.matrices import FormalisationObjects, FormalisationMatrix, principle_formalisation_objs
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
    parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    parser.add_argument('-w', type=int, default=0, help='Weights')
    ## FILE ARGS
    parser.add_argument('-f', type=str, default="value_systems/ESS/PVS_abstracted.csv", help='CSV file with personal values value_systems')
    parser.add_argument('-pf', type=str, default="value_systems/ESS/3q_PriP.csv", help='CSV file with principle value_systems')
    parser.add_argument('-slmf', type=str, default="value_systems/ESS/3q_PriP.csv", help='CSV file with principles for Salas-Molina method SML')
    parser.add_argument("-n_values", type=int, default=4, help='Number of values')
    parser.add_argument("-n_actions", type=int, default=3, help='Number of actions')
    ## COMPUTE ARGS
    parser.add_argument('-hcva', default=False, help='Compute HCVA', action='store_true')
    parser.add_argument('-hcva2', default=False, action='store_true', help='Compute HCVA++')
    parser.add_argument('-slm', default=False, action='store_true', help="Generate consensus using the method described by Salas-Molina et al.")
    parser.add_argument('-t', default=False, help='compute the threshold p, the transition point', action='store_true')
    parser.add_argument('-b', default=False, help='Compute the baseline aggregations (p=1. p=\infty)', action='store_true')
    parser.add_argument('-range', default=False, action='store_true', help='Aggregate everything')
    parser.add_argument('-range_step', type=float, default=0.01, help='Step size for args.range (aggregate everything)')
    # Initialise args and params
    args = parser.parse_args()

    # P_list, J_list, country_dict are the matrices created from the personal values
    # TODO: automatically compute number of actions and values from cols.
    n_values = args.n_values
    n_actions = args.n_actions

    P_list, J_list, w, country_dict = FormalisationObjects(filename=args.f, delimiter=',', weights=args.w, n_values=n_values, n_actions=n_actions)
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
    if args.hcva2:
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
        # Round to 2 d.p. for fairness
        consensus_p = round(consensus_p,2)
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
    if args.slm:
        """ Compute aggregation with Salas-Molina et al. baseline (Many P's) """
        print("Computing SLM")
        # 1. Read in the principles file. Each column contains a set of principles to use.
        file_path = args.slmf
        principles = pd.read_csv(file_path)
        # Convert the principles (which are preferences) into numbers (need to first find transition point
        print("Principles: ", principles)
        #_, _, _, _, transition_p = transition_point(P_list, J_list, w, args.e)
        transition_p = 1.40
        list_of_principles = principles["Egalitarian"].to_list()
        converted_principles = []
        for principle in list_of_principles:
            # Find p by finding the p the relative distance away from the transition point.
            converted_p = pow(transition_p, (2 * principle))
            # Round to 2 d.p. for fairness
            converted_p = round(converted_p, 2)
            converted_principles.append(float(converted_p))
            # TODO: more than 11 values? then it breaks. I'm gonna push this code onto isambard to see what it does
        #converted_principles = np.repeat(1.4, 11)
        print("Converted principles: ", converted_principles)
        # 2. For each list of ps in principles, aggregate and save
        #   there will always be one list, because we aren't testing multiple principle datasets
        now = dt.now().isoformat()
        filename= str("SLM" + "_" + now + ".csv")
        filename_metadata = str("SLM_METADATA_" + now + ".csv")
        p, u_pref, cons_pref = aggregate_slm(P_list, J_list, w, converted_principles, True)
        _, u_act, cons_act = aggregate_slm(P_list, J_list, w, converted_principles, False)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, _, converted_principles, _)
    if args.hcva:
        """ Compute HCVA (closest P/VALE) """
        print("Computing HCVA")
        # 1. Formalise the principle preferences as matrices
        Pri_P_list, _, Pri_w, Pri_Country_dict = principle_formalisation_objs(
            filename=args.pf, delimiter=',', weights=args.w)
        # 2. Aggregate over all principle preferences
        p_list, _, cons_list, _, _, cons_1, cons_l = aggregate_prefs_only(Pri_P_list, [], Pri_w)
        print("Aggregated over all principle preferences!")
        # 3. Find a cutoff point given $\epsilon$
        cut_point = 10
        incr = 0.1
        j = 0
        # This version of epsilon is the same as used in original paper, defined arbitrarily in that case.
        epsilon = 0.05
        for i in np.arange(1 + incr, 10, incr):
            cons = cons_list[j]
            print("cons: ", cons)
            print("cons_1: ", cons_1)
            print("cons_l: ", cons_l)
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            j += 1
            print("dist_1p: ", dist_1p, " dist_pl: ", dist_pl, " i: ", i, "")
            print("abs(dist_1p - dist_pl): ", abs(dist_1p - dist_pl), " epsilon: ", epsilon)
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
    if args.b:
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
                cons_act = cons_act[:len(cons_act) // 2]
            elif p == 1:
                _, u_pref, cons_pref = aggregate_one(P_list, J_list, w, p, True)
                _, u_act, cons_act = aggregate_one(P_list, J_list, w, p, False)
                cons_act = cons_act[:len(cons_act) // 2]
            else:
                # Some other singular p
                _, u_pref, cons_pref = aggregate(P_list, J_list, w, p, True)
                _, u_act, cons_act = aggregate(P_list, J_list, w, p, False)
                cons_act = cons_act[:len(cons_act) // 2]
            output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
            save_metadata(filename_metadata, args, _, p, _)
    if args.range:
        """Aggregate everything between 1-10 + infty with a step size defined as -step_size"""
        p_list, u_list, cons_list, dist1p_list, distpl_list, cons_1, cons_l = aggregate_all_p(P_list, J_list, w, args.range_step)
        # Save to a file
        now = dt.now().isoformat()
        filename = str("COMPLETE_PERSONALS_" + now + ".csv")
        filename_metadata = str("COMPLETE_METADATA_" + now + ".csv")
        _ = None
        output_file(p_list, u_list, cons_list, dist1p_list, distpl_list, cons_1, cons_l, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, _, p_list, _)

