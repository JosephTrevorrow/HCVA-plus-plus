import copy
import pandas as pd
import re
import ast
import os
import glob
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats
import matplotlib

def _source_population_colors(source_data_dir, prip_data_dir, experiment_ids):
    """Return P-based minority/majority labels in residual order."""
    colors_by_timestep = []
    experiment_dirs = [
        os.path.join(source_data_dir, str(experiment_id))
        for experiment_id in experiment_ids
    ]
    for experiment_dir in experiment_dirs:
        experiment_id = os.path.basename(experiment_dir)
        source_files = sort_nicely([
            os.path.join(experiment_dir, entry.name)
            for entry in os.scandir(experiment_dir) if entry.is_file()
        ])
        if not source_files:
            raise ValueError(f"No PVS files found for experiment {experiment_id}")

        for timestep_id, source_file in enumerate(source_files):
            source_values = pd.read_csv(source_file)
            p_columns = [column for column in source_values if column.startswith("P__")]
            if not p_columns:
                raise ValueError(f"PVS file has no P__ columns: {source_file}")
            p_values = source_values[p_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if len(p_values) < 2:
                raise ValueError(f"PVS file must contain at least two agents: {source_file}")
            cluster_ids = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(p_values)
            cluster_sizes = np.bincount(cluster_ids, minlength=2)
            if cluster_sizes[0] == cluster_sizes[1]:
                centers = np.array([
                    p_values[cluster_ids == cluster_id].mean(axis=0)
                    for cluster_id in range(2)
                ])
                majority_cluster = int(np.lexsort(centers.T)[0])
            else:
                majority_cluster = int(np.argmax(cluster_sizes))
            colors = np.where(cluster_ids == majority_cluster, "majority", "minority")
            while len(colors_by_timestep) <= timestep_id:
                colors_by_timestep.append([])
            colors_by_timestep[timestep_id].extend(colors.tolist())
    return colors_by_timestep

def normalise_cons_time_series(sets):
    normalised = {}
    for key, df in sets.items():
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised[key] = copy.deepcopy(df)
    return normalised

def normalise_agents_time_series(sets):
    normalised = []
    for df in sets:
        ## This only works when you know the max and min value of EVERY SINGLE CONS!!!
        # Normalise all data for consensus between 0-1
        #df = df.astype(float)
        for column in df.columns:
            if 'VA__' in column:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
                max_val = 1
            if column != 'p':
                df[column]= (df[column] - min_val) / (max_val - min_val)
        normalised.append(copy.deepcopy(df))
    return normalised

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

def parse_vector_cell(cell):
    """Parse a vector stored in a CSV cell.
    New synthetic consensus files store cons_pref/cons_act in a single
    cell, as a stringified Python-style list.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    if isinstance(cell, str):
        # Remove the whitespace between the "[" and the first character, then the regex will match.
        # Otherwise, you end up with the string "[    mydecimalnumberhere" becoming [,mydecimalnumberhere], which is a syntax error
        cell = cell.strip('[]')
        return ast.literal_eval(','.join(re.sub(r'(?<=\S)(\s+)(?=-?\S)', ',', cell).splitlines()))
    return list(cell)


def plot_synth(agents_pvs_dir, agents_prip_dir, cons_dir, steps, experiment_name, slm_reweighted_dir=None):
    """This method computes plots-synthetic from synthetic data"""
    # Find output_dirs/sum

    ## Something wrong here, things not being read for all runs. we also need to make sure these things are sorted as well -> cons_files_by_dir is not being sorted.
    all_cons_files = os.listdir(cons_dir)
    cons_files_by_dir = {}
    slm_files_by_dir = {}
    list_of_output_dirs = []
    for fname in all_cons_files:
        match = re.search(r"DIR_([0-9]+)_RUN_([0-9]+)", fname)
        if not match:
            continue
        dir_id = match.group(1)
        run_id = match.group(2)
        # Now add the file anyway, even if it isn't an end of line.
        cons_files_by_dir.setdefault(dir_id, [])
        cons_files_by_dir[dir_id].append(os.path.join(cons_dir, fname))
    if slm_reweighted_dir is not None:
        ## Add reweighted SLM if exists.
        if not os.path.isdir(slm_reweighted_dir):
            raise FileNotFoundError(f"SLM reweighted result directory not found: {slm_reweighted_dir}")
        for fname in os.listdir(slm_reweighted_dir):
            match = re.search(r"DIR_([0-9]+)_RUN_([0-9]+)", fname)
            if match:
                slm_files_by_dir.setdefault(match.group(1), []).append(os.path.join(slm_reweighted_dir, fname))
    # Sort: (Note, these are filenames for cons sets)
    for temp_dir_list in cons_files_by_dir.values():
        temp_dir_list = sort_nicely(
            temp_dir_list)  # This says unused, but debug mode shows that it is being sorted, so is being treated as a pointer.

    # Remove all runs that failed to complete (i.e. don't have the right number of timesteps)
    temp_dict = copy.deepcopy(cons_files_by_dir)
    for key, temp_dir_list in temp_dict.items():
        slm_files = slm_files_by_dir.get(key, [])
        if len(temp_dir_list) != int(steps) + 1 or (
                slm_reweighted_dir is not None and len(slm_files) != int(steps) + 1):
            del cons_files_by_dir[key]

    # CAUTION: Cutting for debug
    """
    list_of_output_dirs = list(cons_files_by_dir.keys())[:2]
    temp_dict = {key: cons_files_by_dir[key] for key in list_of_output_dirs}
    cons_files_by_dir = temp_dict
    """

    ## Non-debug version:
    list_of_output_dirs = list(cons_files_by_dir.keys())

    # Construct dir_dict of form {dir_id: [normalised_cons, normalised_agents], ...}
    dir_dict = {}
    for dir in list_of_output_dirs:
        agents_list = []
        # input every single agent, place in a list of df (form: [t1_agents, t2_agents, etc.
        ag_pvs = glob.glob(agents_pvs_dir + str(dir) + "/" + "*PVS*")
        ag_prip = glob.glob(agents_prip_dir + str(dir) + "/" + "*PriP*")
        sorted_ag_pvs = sort_nicely(ag_pvs)
        sorted_ag_prip = sort_nicely(ag_prip)

        # Use the agents header col to name cons_pref and cons_act.
        # These should be stable for all timesteps in the same dir.
        agents_pvs_header_df = pd.read_csv(sorted_ag_pvs[0], nrows=0)
        pref_cols = [col for col in agents_pvs_header_df.columns if 'P__' in col]
        act_cols = [col for col in agents_pvs_header_df.columns if 'VA__' in col]

        # For each cons found, load into a df, store in a dict of dfs with baseline names.
        #   Manually define names as runner.py does not set them
        cons_run_files = sort_nicely(cons_files_by_dir[dir])
        cons_sets_parts = {}
        for run_index, cons_file in enumerate(cons_run_files):
            source_files = [cons_file]
            if slm_reweighted_dir is not None:
                slm_file = sort_nicely(slm_files_by_dir[dir])[run_index]
                source_files.append(slm_file)
            for source_file in source_files:
                cons_run_df = pd.read_csv(source_file)
                if source_file != cons_file:
                    cons_run_df = cons_run_df[cons_run_df["NAME"] == "SLM_REWEIGHTED"]
                for _, row in cons_run_df.iterrows():
                    key = str(row["NAME"])
                    # Make a dataframe after unpacking the values from cons_pref and cons_act, append the Egalitarian pref.
                    cons_df = pd.DataFrame(
                        [[*row[pref_cols], *row[act_cols], row["consensus_preference"]]],
                        columns=pref_cols + act_cols + ["Egalitarian"],
                    )
                    cons_sets_parts.setdefault(key, []).append(cons_df)
        cons_sets = {
            key: pd.concat(parts, ignore_index=True)
            for key, parts in cons_sets_parts.items()
        }
        # Finally, get those agents
        for i in range(0, len(sorted_ag_pvs)):
            agents_pvs_df = pd.read_csv(sorted_ag_pvs[i])
            agents_prip_df = pd.read_csv(sorted_ag_prip[i], usecols=["Egalitarian"])
            agents_df = pd.concat([agents_pvs_df, agents_prip_df], axis=1, join="inner")
            agents_list.append(agents_df)

        # remove the irrelevant cols from every single df you've just sorted out. Create a list of params to use with residuals
        #  Every df will have the same cols, so we find them for one, and copy this
        # note, we will use values_list and actions_list to filter our data analysis plots-synthetic.
        values_list = pref_cols
        cleaned_values_list = values_list.copy()
        # Clean list_of_params
        # Remove all cols that have the same two values (P__Universalism__Universalism, P__Benevolence__Benevolence, etc.)
        for col in values_list:
            col_split = col.split("__")
            if len(col_split) == 3 and col_split[1] == col_split[2]:
                cleaned_values_list.remove(col)
            elif col in cleaned_values_list:
                # Not dropped, so drop the symmetrical col (P__A__B == P__B__A)
                symmetrical_col = "P__" + col_split[2] + "__" + col_split[1]
                if symmetrical_col in cleaned_values_list:
                    cleaned_values_list.remove(symmetrical_col)
        ## Again, because every method will have the exact same col names, we just use HCVA here.
        actions_list = act_cols
        agents_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]
        cons_cols_to_keep = cleaned_values_list + actions_list + ["Egalitarian"]

        # Filter using cols_to_keeps (Agents list and cons list will be the same)
        for i in range(0, len(agents_list)):
            agents_list[i] = agents_list[i][agents_cols_to_keep]
        cons_sets = {k: v[cons_cols_to_keep] for k, v in cons_sets.items()}

        normalised_cons_sets = normalise_cons_time_series(cons_sets)
        normalised_agents_df = normalise_agents_time_series(agents_list)
        dir_dict[int(dir)] = [normalised_cons_sets, normalised_agents_df]
    return dir_dict

def _mean_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = values.mean()
    if values.size == 1:
        return mean, mean
    margin = stats.t.ppf((1 + confidence) / 2, values.size - 1) * stats.sem(values)
    return mean - margin, mean + margin


def _paired_comparison(left, right, method_left, method_right, experiment_id, step):
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError(f"No paired observations for {experiment_id}, step {step}")
    ci_low, ci_high = _mean_confidence_interval(differences)
    difference_std = differences.std(ddof=1) if differences.size > 1 else np.nan
    if np.allclose(differences, 0):
        t_statistic, t_pvalue = np.nan, 1.0
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        t_result = stats.ttest_1samp(differences, 0.0)
        t_statistic, t_pvalue = float(t_result.statistic), float(t_result.pvalue)
        wilcoxon_result = stats.wilcoxon(differences, zero_method="wilcox")
        wilcoxon_statistic, wilcoxon_pvalue = float(wilcoxon_result.statistic), float(wilcoxon_result.pvalue)
    return {
        "experiment": experiment_id,
        "step": step,
        "method_left": method_left,
        "method_right": method_right,
        "n_pairs": int(differences.size),
        "mean_difference_left_minus_right": float(differences.mean()),
        "difference_ci_low": float(ci_low),
        "difference_ci_high": float(ci_high),
        "difference_std": float(difference_std),
        "paired_cohens_d": float(differences.mean() / difference_std) if difference_std > 0 else np.nan,
        "paired_t_statistic": t_statistic,
        "paired_t_pvalue": t_pvalue,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
    }


def _benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    finite = np.isfinite(pvalues)
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        return adjusted
    order = indices[np.argsort(pvalues[indices])]
    ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return adjusted