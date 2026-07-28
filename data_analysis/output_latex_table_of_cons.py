import pandas as pd
import os, glob, copy

def output_latex_table_of_cons(cons_dir, savename, list_of_ids, toFile, toLatex):
    df = pd.DataFrame()
    list_of_filenames = []
    ## for each cons in the cons output_dir, find the personals
    for id in list_of_ids:
        file = glob.glob(cons_dir + id + "*_PERSONALS_*")[0]
        # Assume there is always 1 file, so just pick first
        temp_df = pd.read_csv(file)
        # add a column with the id, and put it at the front of the df
        temp_df.insert(loc=0, column="Method", value=id)
        df = pd.concat([df, temp_df])

    # drop the uninteresting columns from the df and save

    values_list = list([col for col in df.columns if 'P__' in col])
    cleaned_values_list = copy.deepcopy(values_list)
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

    actions_list = list([col for col in df.columns if 'VA__' in col])
    cols_to_keep = ["Method"] + cleaned_values_list + actions_list

    df = df[cols_to_keep]

    # Find the P cols with the largest IQR, keep the top k.
    k = 3
    sorted_IQR_col_names = {}
    for col in cleaned_values_list:
        sorted_IQR_col_names[col] = df[col].quantile(0.75) - df[col].quantile(0.25)
    sorted_IQR_col_names = sorted(sorted_IQR_col_names.items(), key=lambda x: x[1], reverse=True)
    # give the top k
    top_k_cols_ps = [col[0] for col in sorted_IQR_col_names[:k]]

    # Find the VA cols with the largest IQR, keep the top k.
    # Find the P cols with the largest IQR, keep the top k.
    k = 2
    sorted_IQR_col_names = {}
    for col in actions_list:
        sorted_IQR_col_names[col] = df[col].quantile(0.75) - df[col].quantile(0.25)
    sorted_IQR_col_names = sorted(sorted_IQR_col_names.items(), key=lambda x: x[1], reverse=True)
    # give the top k
    top_k_cols_va = [col[0] for col in sorted_IQR_col_names[:k]]

    df = df[["Method"] + top_k_cols_ps + top_k_cols_va]

    # Save the df to a file
    if toFile:
        df.to_csv(savename, index=False)
    elif toLatex:
        print(df.to_latex(index=False,
                          float_format="{:.3f}".format,
                          ))

if __name__ == "__main__":

    cons_dir = "/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/ESS_COUNTRY/4_val_2_acts/"
    list_of_ids = ["HCVApp", "SLM", "T", "HCVA", "1", "inf"]
    savename = "ESS_COUNTRY_4_val_2_acts_table_of_cons.csv"
    output_latex_table_of_cons(cons_dir, savename, list_of_ids, False, True)
