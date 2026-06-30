import argparse as ap
from lp_regression.solve import *
import lp_regression.matrices as matrices
from datetime import date
from files import *
import pandas as pd
import copy as copy
import numpy as np

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-pvs_dir', type=str, help='Directory pointing to the pvs csvs used in the experiment')
    parser.add_argument('-prip_dir', type=str, help='Directory pointing to the prips csvs used in the experiment')
    ## ENV ARGS
    parser.add_argument("-n_values", nargs="*", type=int, default=[4], help='Number of values')
    parser.add_argument("-n_actions",nargs="*", type=int, default=[3], help='Number of actions')
    parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    parser.add_argument('-w', type=int, default=0, help='Weights')

    parser.add_argument('-output_dir', type=str, default="output", help='Directory to save the output files')
    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.

    # Initialise args and params
    args = parser.parse_args()
    # Note, these are lists
    n_values = args.n_values
    n_actions = args.n_actions
    output_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus" + args.output_dir
    now = str(date.today())
    print(now)

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
        # Update the num_vals and num_actions if necessary
        if i < len(n_values):
            n_values = n_values[i]
        if i < len(n_actions):
            n_actions = n_actions[i]

        # Preprocess the csvs
        print("My pvs set is: ", pvs_sets[i], " and my prip set is: ", prip_sets[i], "")
        ## PVS
        print("PREPROCESSING PVS...")
        P_list, J_list, w, country_dict = FormalisationObjects(filename=pvs_sets[i], delimiter=',', weights=args.w,
                                                               n_values=n_values, n_actions=n_actions)
        pvs_df = pd.read_csv(pvs)
        ### Below is only used for col headings when saving to a file
        values_list = list([col for col in pvs_df.columns if 'P__' in col])
        actions_list = list([col for col in pvs_df.columns if 'VA__' in col])
        ## PriPs
        prip_df = pd.read_csv(prip_sets[i])
        print("-------------")

        # Do a full run of aggregations and then cache it, to speed up computation

        # Run aggregations
        ## SLM
        print("SLM...")
        filename = str("SLM_PERSONALS_RUN_"+str(i)+"_"+ now + ".csv")
        filename_metadata = str("SLM_METADATA_"+str(i)+"_"+ now + ".csv")
        p, u_pref, cons_pref, u_act, cons_act, transition_p, converted_principles = find_slm_and_aggregate(P_list, J_list, w, prip, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
        save_metadata(filename_metadata, args, 0, converted_principles, 0, output_dir)
        print("-------------")
        ## HCVA++
        print("HCVA++...")
        filename = str("HCVApp_PERSONALS_" +str(i)+"_"+now + ".csv")
        filename_metadata = str("HCVApp_METADATA_" +str(i)+"_"+now + ".csv")
        p, u_pref, cons_pref, u_act, cons_act, consensus_p, transition_p, consensus_preference = find_hcva_pp_and_aggregate(P_list, J_list, w, prip, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
        save_metadata(filename_metadata, args, transition_p, consensus_p, consensus_preference, output_dir)
        print("-------------")
        ## EGAL/UTIL
        print("EGAL/UTIL...")
        baseline_ps = [1, np.inf]
        for p in baseline_ps:
            # Generate filenames
            filename = str(str(p) + "_PERSONALS_" +str(i)+"_"+ now + ".csv")
            filename_metadata = str(str(p) + "_METADATA_" +str(i)+"_"+ now + ".csv")
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
            output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            save_metadata(filename_metadata, args, 0, p, p, output_dir)
        print("-------------")
        ## T
        print("T...")
        filename = str("T_PERSONALS_" +str(i)+"_"+ now + ".csv")
        filename_metadata = str("T_METADATA_" +str(i)+"_"+ now + ".csv")
        filename_limits = now + "_limits.csv"
        p, u_pref, cons_pref, u_act, cons_act, t_point = find_transition_and_aggregate(P_list, J_list, w,
                                                                                       filename_limits, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
        # t point = 0.5 preference
        save_metadata(filename_metadata, args, t_point, t_point, 0.5, output_dir)
        print("-------------")

        ## HCVA
        print("HCVA...")
        filename = str("HCVA_PERSONALS_"+str(i)+"_"+now + ".csv")
        filename_metadata = str("HCVA_METADATA_"+str(i)+"_"+now + ".csv")
        p, u_pref, u_act, cons_pref, cons_act, con_p = find_hcva_and_aggregate(P_list, J_list, w, prip_df, args)
        output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
        # Convert HCVA to a preference.
        con_preference = np.log(con_p) / (2*np.log(t_point))
        save_metadata(filename_metadata, args, None, con_p, None, output_dir)
        print("-------------")
        print("Done with iteration: {}".format(i))
