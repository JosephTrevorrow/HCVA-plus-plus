import argparse as ap
from lp_regression.solve import *
import lp_regression.matrices as matrices
from datetime import datetime as dt
from files import *
import pandas as pd
import copy as copy

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-pvs_dir', type=str, help='Directory pointing to the pvs csvs used in the experiment')
    parser.add_argument('-prip_dir', type=str, help='Directory pointing to the prips csvs used in the experiment')
    ## ENV ARGS
    parser.add_argument("-n_values", nargs="*", type=int, default=[4], help='Number of values')
    parser.add_argument("-n_actions",nargs="*", type=int, default=3, help='Number of actions')
    parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.

    # Initialise args and params
    args = parser.parse_args()
    n_values = args.n_values
    n_actions = args.n_actions

    now = dt.now().isoformat()

    ## Begin by finding a list of all pvs and prips
    pvs_sets = []
    pvs_dir = args.pvs_dir
    for file in os.listdir(pvs_dir):
        if file.endswith(".csv"):
            pvs_sets.append(pvs_dir + file)
    len_of_pvs_sets = len(pvs_sets)
    pvs = pvs_sets[0]

    prip_sets = []
    prip_dir = args.prip_dir
    for file in os.listdir(prip_dir):
        if file.endswith(".csv"):
            prip_sets.append(prip_dir + file)
    len_of_prip_sets = len(prip_sets)
    prip = prip_sets[0]

    ## Big for loop here running each aggregation and then saving!
    # For the max of prip or pvs sets,
    for i in range(0, max(len(pvs_sets), len(prip_sets))):
        print("Iteration: {}".format(max(len(pvs_sets), len(prip_sets))))
        ## update the pvs and prip
        if i < len(pvs_sets):
            print("PVS: ", i)
            pvs = pvs_sets[i]
        if i < len(prip_sets):
            print("PriP: ", i)
            prip = prip_sets[i]

        # Preprocess the csvs
        ## PVS
        P_list, J_list, w, country_dict = FormalisationObjects(filename=pvs_sets[i], delimiter=',', weights=args.w,
                                                               n_values=n_values, n_actions=n_actions)
        pvs_df = pd.read_csv(pvs)
        ### Below is only used for col headings when saving to a file
        values_list = list([col for col in pvs_df.columns if 'P__' in col])
        actions_list = list([col for col in pvs_df.columns if 'VA__' in col])
        ## PriPs
        prip_df = pd.read_csv(prip_sets[i])

        # Run aggregations
        ## SLM
        print("SLM...")
        filename = str("SLM" + "_" + now + ".csv")
        filename_metadata = str("SLM_METADATA_" + now + ".csv")
        p, u_pref, cons_pref, u_act, cons_act, transition_p, converted_principles = find_slm_and_aggregate(P_list, J_list, w, prip, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, _, converted_principles, None)

        ## HCVA++
        print("HCVA++...")
        filename = str("HCVApp_" + now + ".csv")
        filename_metadata = str("HCVApp_METADATA_" + now + ".csv")
        p, u_pref, cons_pref, u_act, cons_act, consensus_p, transition_p, consensus_preference = find_hcva_pp_and_aggregate(P_list, J_list, w, prip, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, transition_p, consensus_p, consensus_preference)

        ## EGAL/UTIL
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

        ## T
        print("T...")
        filename = str("T_" + now + ".csv")
        filename_metadata = str("T_METADATA_" + now + ".csv")
        filename_limits = now + "_limits.csv"
        p, u_pref, cons_pref, u_act, cons_act, t_point = find_transition_and_aggregate(P_list, J_list, w,
                                                                                       filename_limits, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list)
        save_metadata(filename_metadata, args, t_point, None, None)

        ## HCVA
