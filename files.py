import csv

def save_metadata(filename, args, transition_p=None, consensus_p=None, consensus_preference=None):
    # TODO: deal with inputs not given
    # Save this properly
    csv_rows = [{"args":args, "transition_p":transition_p, "consensus_p":consensus_p, "consensus_preference":consensus_preference}]
    with open(filename, 'w', newline='') as csvfile:
        # writing file
        fieldnames = ["args", "transition_p", "consensus_p", "consensus_preference"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return None

def output_single(p, u_pref, u_act, cons_pref, cons_act, filename, values_list, actions_list):
    """
    This function writes the results of solving the lp-regression for a single p
    This is usually the output of a single experiment.
    """
    header = ['p', 'U_pref', 'u_act', 'dist_1', 'dist_l'] + values_list + actions_list
    csv_rows = [header]
    # Baselines # TODO: dist_l and dist_1 are placeholders
    row = [p, u_pref, u_act, 0, 0]
    # Write cons
    row = row + list(cons_pref) + list(cons_act)
    # Write to a file
    csv_rows.append(row)
    with open(filename, 'w', newline='') as csvfile:
        # writing file
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    csvfile.close()
    return None

def output_file(p_list, U_list, cons_list, dist_1p_list, dist_lp_list, cons_1, cons_l, filename,values_list, actions_list):
    """
    This function writes the results of solving the lp-regression for different p's.
    INPUT: p -- int, U -- list (Up distance function values), cons -- list of list
           (values of the consensus achieved per p), dist_1 -- list (distance from
           the consensus achieved for p=1 and the one for current p), dist_l -- list
           (distance from the consensus achieved for p=inf and the one for current p),
           v -- boolean (parameter, when v = True, we solve the prefference aggregation
           over moral values, when v = False, we solve the aggregation of moral values),
           name -- str (name of the file)
    """
    csv_rows = []
    header = ['p', 'U_pref', 'u_act', 'dist_1', 'dist_l'] + values_list + actions_list
    csv_rows.append(header)
    for i in range(len(p_list)):
        el = [p_list[i], U_list[i][0], U_list[i][1], dist_1p_list[i], dist_lp_list[i]]
        el.extend(cons_list[i])
        csv_rows.append(el)
    with open(filename, 'w', newline='') as csvfile:
        # writing file
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    csvfile.close()
    return None

def simple_output_file(p, y, name):
    """
    This function writes a measure for each p in a file.
    INPUT: p -- int, y -- list (measure), name -- str (name of the file)
    """
    header = ["p", "y(p)"]
    csv_rows = [header]
    for i in range(len(p)):
        el = [p[i], y[i]]
        csv_rows.append(el)
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    csvfile.close()
    return None

def limit_output(p, dist_p_list, dist_inf_list, diff_list, name):
    """
    This function writes the values computed finding the limit P, and transition points (-t True)
    INPUT: p -- int, y -- list (measure), name -- str (name of the file)
    """
    header = ["p", "Dist_p", "Dist_inf", "Diff"]
    csv_rows = [header]
    for i in range(len(p)):
        el = [p[i], dist_p_list[i], dist_inf_list[i], diff_list[i]]
        csv_rows.append(el)
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    csvfile.close() 
    return None