import argparse as ap
from lp_regression.solve import *
import lp_regression.matrices as matrices
from datetime import date
from files import *
import pandas as pd
import traceback
import copy as copy
import numpy as np
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import PyCall
import re
import multiprocessing as mp

_JL_MAIN = None  # set once per worker by _worker_init

def _worker_init():
    """Runs once when each worker process starts."""
    global _JL_MAIN
    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main

    action_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'lp_regression/IRLS-pNorm.jl')
    )
    Main.eval(f'include("{action_path}")')
    Main.eval("using Main.MyActionModule")
    _JL_MAIN = Main


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

def run_experiment(task):
    (args, current_dir, i, now, output_dir, pvs_sets, prip_sets, n_values, n_actions) = task
    tag = f"[{current_dir} / run {i}]"
    try:
        print(f"{tag} PREPROCESSING PVS...")
        ## PVS
        print("PREPROCESSING PVS...")
        P_list, J_list, w, country_dict = FormalisationObjects(filename=pvs_sets[i], delimiter=',', weights=args.w,
                                                               n_values=n_values, n_actions=n_actions)
        pvs_df = pd.read_csv(pvs_sets[0])
        ### Below is only used for col headings when saving to a file
        values_list = list([col for col in pvs_df.columns if 'P__' in col])
        actions_list = list([col for col in pvs_df.columns if 'VA__' in col])
        ## PriPs
        prip_df = pd.read_csv(prip_sets[i])
        filename = f"DIR_{current_dir}_RUN_{i}_{now}.csv"
        rows = []
        # Run aggregations
        ## T
        # We do T first because we will use the t_point for other methods
        filename_limits = "LIMITS_DIR_" + str(current_dir) + "_RUN_" + str(i) + "_" + now + "limits.csv"
        p, u_pref, cons_pref, u_act, cons_act, t_point = find_transition_and_aggregate(P_list, J_list, w,
                                                                                       output_dir, filename_limits,
                                                                                       args.e, args)
        rows.append(["T", p, u_pref, u_act, cons_pref, cons_act, t_point, t_point, 0.5])
        ## SLM
        p, _, cons_pref, _, cons_act, converted_principles = find_slm_and_aggregate(P_list, J_list, w, prip_df, t_point,
                                                                                    args)
        ## Chop off half of cons_act, as output format has VA_p, and then VA_n. We are not interested in N, so we disregard
        cons_act = cons_act[:len(cons_act) // 2]
        rows.append(["SLM", p, _, _, cons_pref, cons_act, 0, converted_principles, 0, ])
        ## HCVA
        p, u_pref, u_act, cons_pref, cons_act, con_p = find_hcva_and_aggregate(P_list, J_list, w, prip_df, args)
        # output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
        # Convert HCVA to a preference.
        con_preference = np.log(con_p) / (2 * np.log(t_point))
        rows.append(["HCVA", p, u_pref, u_act, cons_pref, cons_act, None, con_p, con_preference])
        ## HCVA++
        p, u_pref, cons_pref, u_act, cons_act, consensus_p, transition_p, consensus_preference = find_hcva_pp_and_aggregate(
            P_list, J_list, w, prip_df, t_point, args)
        rows.append(["HCVA++", p, u_pref, u_act, cons_pref, cons_act, transition_p, consensus_p, consensus_preference])
        ## EGAL/UTIL
        baseline_ps = [1, np.inf]
        for p in baseline_ps:
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
            # output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list, output_dir)
            if p == np.inf:
                rows.append(["inf", p, u_pref, u_act, cons_pref, cons_act, 0, p, 1])
                # save_metadata(filename_metadata, args, 0, p, 1, output_dir)
            elif p == 1:
                rows.append(["1", p, u_pref, u_act, cons_pref, cons_act, 0, p, 0])
                # save_metadata(filename_metadata, args, 0, p, 0, output_dir)
            else:
                rows.append(["1", p, u_pref, u_act, cons_pref, cons_act, 0, p, p])
                # save_metadata(filename_metadata, args, 0, p, p, output_dir)
        # Save the rows to a csv.
        header = ['p', 'U_pref', 'u_act', ] + values_list + actions_list + ['transition_p', 'consensus_p',
                                                                            'consensus_preference']
        with open(output_dir + filename, 'w', newline='') as csvfile:
            # writing file
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        csvfile.close()
        print(f"{tag} done -> {output_dir+filename}")
        return (current_dir, i, True, None)
    except Exception as e:
        print(f"{tag} failed -> {e}")
        return (current_dir, i, False, traceback.format_exc())

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
    parser.add_argument('-n_workers', type=int, default=None,
                         help='Worker pool size. Defaults to $SLURM_CPUS_PER_TASK, then os.cpu_count().')

    # Looking for the number of agents? This is not explicitly defined and can be found from the corresponding pvs_dir and prip_dir of each experiment.
    args = parser.parse_args()

    # Note, these are lists
    n_values_list = args.n_values
    n_actions_list = args.n_actions
    output_dir = args.output_dir
    now = str(date.today())
    print(now)
    os.makedirs(output_dir, exist_ok=True)
    all_dirs = sort_nicely(os.listdir(args.pvs_dir))[args.min:args.max]

    tasks = []
    for idx, current_dir in enumerate(all_dirs):
        if not os.path.isdir(args.pvs_dir + current_dir):
            continue

        pvs_dir = args.pvs_dir + current_dir + "/"
        pvs_sets = sort_nicely([pvs_dir + f for f in os.listdir(pvs_dir) if f.endswith(".csv")])

        prip_dir = args.prip_dir + current_dir + "/"
        prip_sets = sort_nicely([prip_dir + f for f in os.listdir(prip_dir) if f.endswith(".csv")])

        if len(pvs_sets) != len(prip_sets):
            print(f"WARNING: {current_dir} has {len(pvs_sets)} pvs set(s) but "
                  f"{len(prip_sets)} prip set(s) — only the first "
                  f"{min(len(pvs_sets), len(prip_sets))} will be run.")

        n_values = n_values_list[idx] if idx < len(n_values_list) else n_values_list[-1]
        n_actions = n_actions_list[idx] if idx < len(n_actions_list) else n_actions_list[-1]

        for i in range(min(len(pvs_sets), len(prip_sets))):
            tasks.append((args, current_dir, i, now, output_dir,
                          pvs_sets, prip_sets, n_values, n_actions))

    n_workers = args.n_workers or int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Running {len(tasks)} task(s) across {n_workers} worker process(es)")

    # 'spawn' (not the Linux default 'fork') is required: each worker boots
    # its own independent Julia runtime in _worker_init, and forking a
    # process that already has Julia loaded is unsupported.
    ctx = mp.get_context('spawn')
    failures = []
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for current_dir, i, ok, err in pool.imap_unordered(run_experiment, tasks):
            if not ok:
                failures.append((current_dir, i, err))
                print(f"FAILED: {current_dir} run {i}\n{err}")

    print(f"Finished. {len(tasks) - len(failures)}/{len(tasks)} succeeded.")