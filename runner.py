import argparse as ap
from lp_regression.solve import *
import lp_regression.matrices as matrices
from datetime import date
from files import *
import pandas as pd
import copy as copy
import numpy as np
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import PyCall
import re

## CREDIT: https://nedbatchelder.com/blog/200712/human_sorting
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    ## FILE ARGS
    parser.add_argument('-pvs_dir', type=str, help='Directory pointing to the pvs csvs used in the experiment')
    parser.add_argument('-prip_dir', type=str, help='Directory pointing to the prips csvs used in the experiment')
    ## ENV ARGS
    parser.add_argument("-n_values", nargs="*", type=int, default=[4], help='Number of values')
    parser.add_argument("-n_actions",nargs="*", type=int, default=[2], help='Number of actions')
    parser.add_argument('-e', type=float, default=1e-4, help='Epsilon cut-point for T')
    parser.add_argument('-w', type=int, default=0, help='Weights')

    parser.add_argument('-max', type=int, default=40, help='maximum dir to search')
    parser.add_argument('-min', type=int, default=0, help='minimum dir to search')


    parser.add_argument('-output_dir', type=str, default="output", help='Directory to save the output files')
    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.

    # Initialise args and params
    args = parser.parse_args()
    # Note, these are lists
    n_values_list = args.n_values
    n_actions_list = args.n_actions
    output_dir = args.output_dir
    now = str(date.today())
    print(now)

    ## Boot up julia
    ## HPC
    action_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'lp_regression/IRLS-pNorm.jl')
    )
    jl.eval(f'include("{action_path}")')
    jl.eval("""using Main.MyActionModule""")
    max_dir = args.max
    min_dir = args.min
    # Because we are going to run many experiments, we put this in a big for loop
    # find all the dirs in pvs (will be same for prip)
    all_dirs = sort_nicely(os.listdir(args.pvs_dir))
    # Cut based on our min/max
    all_dirs = all_dirs[min_dir:max_dir]
    for current_dir in all_dirs:
        if os.path.isdir(args.pvs_dir+current_dir):
            print("Found output_dir: ", args.pvs_dir+current_dir)
        else:
            # is not a subdirectory? then skip
            continue

        ## Begin by finding a list of all pvs and prips
        pvs_sets = []
        pvs_dir = args.pvs_dir+current_dir+"/"
        for file in os.listdir(pvs_dir):
            if file.endswith(".csv"):
                pvs_sets.append(pvs_dir + file)
        len_of_pvs_sets = len(pvs_sets)

        pvs_sets = sort_nicely(pvs_sets)
        pvs = pvs_sets[0]

        prip_sets = []
        prip_dir = args.prip_dir+current_dir+"/"
        for file in os.listdir(prip_dir):
            if file.endswith(".csv"):
                prip_sets.append(prip_dir + file)
        len_of_prip_sets = len(prip_sets)
        prip_sets = sort_nicely(prip_sets)

        ## Big for loop here running each aggregation and then saving
        # For the max of PriP or pvs sets,
        # This for loop assumes the pvs_sets and prip sets have the exact same len.
        for i in range(0, max(len(pvs_sets), len(prip_sets))):
            print("Iteration: ",i, " of ", max(len(pvs_sets), len(prip_sets)),)
            # Update the num_vals and num_actions if necessary
            if i < len(n_values_list):
                n_values = n_values_list[i]
            if i < len(n_actions_list):
                n_actions = n_actions_list[i]

            # Preprocess the csvs

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

            filename = str("DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
            rows = []
            # Do a full run of aggregations and then cache it, to speed up computation

            # Run aggregations

            ## T
            # We do T first because we will use the t_point for other methods
            print("T...")
            #filename = str("T_PERSONALS_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
            #filename_metadata = str("T_METADATA_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
            filename_limits = "LIMITS_DIR_"+str(current_dir)+"_RUN_"+ str(i)+ "_"+now+"limits.csv"
            p, u_pref, cons_pref, u_act, cons_act, t_point = find_transition_and_aggregate(P_list, J_list, w,
                                                                                           output_dir, filename_limits, args.e, args)
            rows.append(["T", p, u_pref, u_act, cons_pref, cons_act, t_point, t_point, 0.5])
            #output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            #save_metadata(filename_metadata, args, t_point, t_point, 0.5, output_dir)
            print("-------------")
            ## SLM
            print("SLM...")
            #filename = str("SLM_PERSONALS_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
            #filename_metadata = str("SLM_METADATA_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
            p, _, cons_pref, _, cons_act, converted_principles = find_slm_and_aggregate(P_list, J_list, w, prip_df, t_point, args)
            ## Chop off half of cons_act, as output format has VA_p, and then VA_n. We are not interested in N, so we disregard
            cons_act = cons_act[:len(cons_act) // 2]
            rows.append(["SLM", p, _, _, cons_pref, cons_act,0, converted_principles, 0,])
            #output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            #save_metadata(filename_metadata, args, 0, converted_principles, 0, output_dir)
            print("-------------")
            ## HCVA
            print("HCVA...")
            #filename = str("HCVA_PERSONALS_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+now + ".csv")
            #filename_metadata = str("HCVA_METADATA_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+now + ".csv")
            p, u_pref, u_act, cons_pref, cons_act, con_p = find_hcva_and_aggregate(P_list, J_list, w, prip_df, args)
            #output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            # Convert HCVA to a preference.
            con_preference = np.log(con_p) / (2*np.log(t_point))
            #save_metadata(filename_metadata, args, None, con_p, con_preference, output_dir)
            rows.append(["HCVA", p, u_pref, u_act, cons_pref, cons_act, None, con_p, con_preference])
            print("-------------")
            ## HCVA++
            print("HCVA++...")
            #filename = str("HCVApp_PERSONALS_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+now + ".csv")
            #filename_metadata = str("HCVApp_METADATA_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+now + ".csv")
            p, u_pref, cons_pref, u_act, cons_act, consensus_p, transition_p, consensus_preference = find_hcva_pp_and_aggregate(P_list, J_list, w, prip_df, t_point, args)
            #output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            #save_metadata(filename_metadata, args, transition_p, consensus_p, consensus_preference, output_dir)
            rows.append(["HCVA++", p, u_pref, u_act, cons_pref, cons_act, transition_p, consensus_p, consensus_preference])
            print("-------------")
            ## EGAL/UTIL
            print("EGAL/UTIL...")
            baseline_ps = [1, np.inf]
            for p in baseline_ps:
                # Generate filenames
                #filename = str(str(p) + "_PERSONALS_DIR_"+str(current_dir)+"_RUN_"+ str(i)+"_"+now + ".csv")
                #filename_metadata = str(str(p) + "_METADATA_DIR_"+str(current_dir)+"_RUN_"+str(i)+"_"+ now + ".csv")
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
                #output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
                if p == np.inf:
                    rows.append(["inf", p, u_pref, u_act, cons_pref, cons_act, 0, p, 1])
                    #save_metadata(filename_metadata, args, 0, p, 1, output_dir)
                elif p == 1:
                    rows.append(["1", p, u_pref, u_act, cons_pref, cons_act, 0, p, 0])
                    #save_metadata(filename_metadata, args, 0, p, 0, output_dir)
                else:
                    rows.append(["1", p, u_pref, u_act, cons_pref, cons_act, 0, p, p])
                    #save_metadata(filename_metadata, args, 0, p, p, output_dir)
            # Save the rows to a csv.
            header = ['p', 'U_pref', 'u_act', ] + values_list + actions_list + ['transition_p', 'consensus_p', 'consensus_preference']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("Created directory: " + output_dir)
            else:
                print("Directory already exists: " + output_dir)
            with open(output_dir + filename, 'w', newline='') as csvfile:
                # writing file
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            csvfile.close()
            print("-------------")
            print("Done with iteration: {}".format(i))


