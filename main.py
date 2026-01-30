import argparse as ap
import numpy as np
import os
import csv
import copy
from datetime import datetime as dt
from lp_regression.matrices import FormalisationObjects, FormalisationMatrix
from lp_regression.solve import L1, L2, Linf, Lp, mLp, transition_point, aggregate
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
        default="22-01-2025-agent-data.csv",
        help='CSV file with personal values data')
    parser.add_argument(
        '-pf',
        type=str,
        default="input_data/principles/placeholder_principles.csv",
        help='CSV file with principle data')
    parser.add_argument(
        '-smlf',
        type=str,
        default="input_data/sml_principles/placeolder_sml.csv",
        help='CSV file with data for Salas-Molina method SML'
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
    p = args.p
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
        Compute HCVA (mean)
        """
        # 1. Find the consensus priciple $p$
        # 1.1 Find the consensus principle preference
        principle_preferences = []
        with open(args.pf) as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # get rid of header row
            for row in reader:
                temp_prefernce = float(row[1])
                principle_preferences.append(copy.copy(temp_prefernce))
        consensus_preference = sum(principle_preferences) / len(principle_preferences)
        print("Consensus preference is: ", consensus_preference)
        consensus_preference = round(consensus_preference,2)
        # 1.2 Aggregate personal values/action judgements to find the transition point
        _, _, _, _, transition_p =  transition_point(P_list, J_list, w, args.v, args.e, p)
        
        # 1.3 Given the transition point (best_p), find the consensus p by finding the
        # p the relative distance away from the transition point.
        consensus_p = pow(transition_p, (2*consensus_preference))

        # 1.4 Aggregate all of the principle preference values submitted by agents 
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
        Compute aggregation with Salas-Molina et al. baseline
        """


        