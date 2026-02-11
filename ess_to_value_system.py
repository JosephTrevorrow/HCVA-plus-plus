import pandas as pd
import csv
import copy
import numpy as np
from datetime import datetime as dt

def process_all_country_values(ess_df, country_col_name, values_dict, higher_order_values_index_list, abstract):
    """
    Docstring for process_all_country
    
    :param ess_df: The dataframe containing all ESS value_systems
    :param country_col_name: typically 'cntry', the col that will state which country each row belongs to
    :param values_dict: Dict of form str(value) : [question_id's that relate to each value]
    """
    # Find the list of all countries
    ess_country_list = ess_df[country_col_name].unique()

    # Iterate through each country and find their values
    country_values = {}
    for country in ess_country_list:
        country_df = ess_df.loc[ess_df[country_col_name] == country].copy()
        temp_values = []
        # Iterate through and find all values, summing for each 
        #   question. temp_values will be in same order as values_dict
        # Then find the central preference by finding mean of each of these.

        # For principles only: We keep the iteration through the dict but note that for standard L_p regression there is only
        # one P, with 3 questions.
        for _, values in values_dict.items():
            temp_questions = []
            for question in values:
                # Invert items such that higher scores represent greater importance (inverted = 1 = worst, 6 = best)
                if question == "sofwrk":
                    country_df[question] = country_df[question]
                else:
                    country_df[question] = 7 - country_df[question]
                # Find mean_score for every col
                mean_score = country_df[question].sum() / len(country_df)
                # temp_questions contains the mean score of every question for that specific value, in the same
                #   order as the list for the values_dict
                temp_questions.append(mean_score)
            # temp_values contains the list of mean scores for each question for that value, in the same order as the values_dict
            temp_values.append(temp_questions)
        # country_values contains the list of mean scores for every question for the country
        country_values[country] = temp_values

    # Subtract the mean score across all values (for a single country) from a value's raw score. This produces centered values.
    # country_values = dict({country_name: [[value1_q1, value1_q2], [value2_q1, value2_q2], [..]]})
    for country, values in country_values.items():
        # Find the sum of all the values in the nested list (values) and compute the mean
        total = 0
        num_of_questions = 0
        for value_list in values:
            num_of_questions += len(value_list)
            total += sum(value_list)
        mean = total / num_of_questions

        # For every value in the nested lists: values, subtract the mean from it, to find a
        #  centered value for that question. Then find the mean of that list
        for value_list in values:
            for value in value_list:
                value_list[value_list.index(value)] = value - mean
        temp_country_vals = np.array([np.mean(v) for v in country_values[country]])
        country_values[country] = copy.copy(temp_country_vals)
        ## Now we have a list of centred country_values for every country, if we want to convert these to higher order values
        # then we find the mean of these.
        if abstract:
            print("Abstracting: ", country_values[country])
            abstracted_values = np.array([])
            for h_vals_list in higher_order_values_index_list:
                temp_sum = 0
                for index in h_vals_list:
                    temp_sum += country_values[country][index]
                mean = temp_sum / len(h_vals_list)
                abstracted_values = np.append(abstracted_values, [mean])
            country_values[country] = copy.copy(abstracted_values)
            print("Abstracted: ", country_values[country])

    # we find the preferences between each value and every other value, and store
    value_preferences = {}
    for country, values_list in country_values.items():
        print("type of values_list: ", type(values_list))
        print("values list: ", values_list)
        # Principle case: There is only one value, so just store
        if len(values_list) == 1:
            value_preferences[country] = values_list[0]
            continue
        # For every value in the values list, compare it to every other value
        diff = values_list[:, np.newaxis] - values_list
        diff_norm = (diff- np.min(diff)) / (np.max(diff) - np.min(diff))
        value_preferences[country] = copy.copy(diff_norm)
        """
        array([1, 4, 6])
        [0,0], [0, 1]
        array([[ 0, -3, -5],
               [ 3,  0, -2],
               [ 5,  2,  0]])

        array([uni, ben, tra, con, sec, pow, ach, hed, sti, sel])
        array([[ [uni, uni], [uni, ben], [uni, tra]],
               [ [ben, uni], [ben, ben], [ben, tra]],
               [ [tra, uni],  [tra,ben],  [tra, tra]]])
        """
    return value_preferences
    
def process_all_country_actions(ess_df, country_col_name, value_preferences, actions_dict):
    """
    Given a country's value preferences, find their action value judgements
    INPUT: ess_df 
        country_col_name
        value_preferences: NxN np.array, comparing each value to every other value
        actions_dict: dict of actions in form str(action_name) : question_id
    """
    # Find the list of all countries
    ess_country_list = ess_df[country_col_name].unique()

    # First, iterate through each country and find their centred actions (between -1 and 1)
    country_centred_actions = {}
    for country in ess_country_list:
        country_df = ess_df.loc[ess_df[country_col_name] == country].copy()
        temp_actions = []
        # Iterate through and find all values, summing for each 
        #   question. temp_values will be in the same order as values_dict
        # Then find the central preference by finding mean of each of these.
        for _, action_id in actions_dict.items():
            # No need to invert immigration action, as score between 1-10, where 10 is GOOD, 1 is BAD,
            # brexit is binary, so either 1-2. Order doesn't matter, higher will mean leave as the question marks 2 as leave.
            if action_id == "freehms" or action_id == "hmsacld":
                country_df[action_id] = 6 - country_df[action_id]
            # Find the mean score for the action (1 action, no need to find average of a set)
            mean = country_df[action_id].mean()
            mean_score = mean.iloc[0]
            # Centre the scores between -1 and 1. action_id is always a list of size 1
            # immigration is between 1-11, where 11 is GOOD, 1 is BAD, other actions are between 1 (good) - 5 (bad)
            if action_id[0] == "imbgeco":
                # Because immigration is scored is between 1-11
                centred_score = (((mean_score-1) * 2)/9)
            elif action_id[0] == "vteumbgb":
                # centre between -1 and 1, for values being either 1 or 2
                centred_score = (mean_score * 2) - 3
            else:
                # Actions other than "imbgeco" and "vteumgb" are between 1-5, where 1 = agree strongly,
                # and 5 = disagree strongly.
                centred_score = (((mean_score-1) * 2)/4) - 1
            temp_actions.append(float(centred_score))
        # country_values contains the averaged, centred action for each action
        country_centred_actions[country] = np.array(temp_actions, dtype=float)

    # Second, convert country actions into action judgements considering their value preferences
    action_judgements = {}
    for country in ess_country_list:
        # country_values shape is [x,x], x= num of values
        country_values = np.array(value_preferences[country], dtype=float)
        x = country_values.shape[0]
        # Convert the preferences to range [-1,1], where shifted_prefs[x,y] >0 means x preferred to y
        shifted_prefs = 2.0 * (country_values - 0.5)
        # Convert the shifted prefs to a symmetric matrix, and
        #   then show preference relative to every other value. as a 1D array
        np.fill_diagonal(shifted_prefs, 0.0)
        # The clipping below should protect against divide by 0 cases, if any arise.
        strengths_of_prefs = shifted_prefs.sum(axis=1) / float(x-1)
        strengths_of_prefs = np.clip(strengths_of_prefs, -1.0, 1.0)
        # Get the centred actions computed above and clip for sanity
        centred_actions = country_centred_actions[country]
        centred_actions = np.clip(centred_actions, -1.0, 1.0)
        ## Finally, convert the centred actions to judgements by multiplying by the strength
        judgements = np.outer(strengths_of_prefs, centred_actions)
        judgements = np.clip(judgements, -1.0, 1.0)
        # Final sanity check
        for judgement in judgements:
            if judgement is None:
                print("Error: judgement is None")
        action_judgements[country] = judgements
    return action_judgements

if __name__ == '__main__':
    df = pd.read_csv("ESS9-private/ESS9e03_2.csv")
    higher_order_values_dict = {
        "Self-Transcendence" : ["Universalism", "Benevolence"],
        "Conservation" : ["Tradition", "Conformity", "Security"], 
        "Self-Enhancement" : ["Power", "Achievement", "Hedonism"], 
        "Openness to Change" : ["Hedonism", "Stimulation", "Self-Direction"]
    }
    higher_order_values_index_list = [[0,1], [2,3,4], [5,6,7], [7,8,9]]

    values_dict = {
                "Universalism": ["ipeqopt", "ipudrst", "impenv"], 
                "Benevolence": ["iphlppl", "iplylfr"],

                "Tradition": ["ipmodst", "imptrad"], 
                "Conformity": ["ipfrule", "ipbhprp"],
                "Security": ["impsafe", "ipstrgv"],

                "Power": ["imprich", "iprspot"], 
                "Achievement": ["ipshabt", "ipsuces"], 
                "Hedonism": ["ipgdtim", "impfun"], 

                "Stimulation": ["impdiff", "ipadvnt"],
                "Self-Direction": ["ipcrtiv", "impfree"],  
    }
    # Principles are different than values, as instead of like me/not like me, it is agree/disagree.
    principle_dict = {
        "Egalitarian" : ["sofrdst"],
    }
    """ "sofrdst" 	Society fair when income and wealth is equally distributed
    1 	Agree strongly
    2 	Agree
    3 	Neither agree nor disagree
    4 	Disagree
    5 	Disagree strongly
    7 	Refusal*
    8 	Don't know*
    9 	No answer*
    "sofrpr" 	Society fair when takes care of poor and in need, regardless of what give back"
    1 	Agree strongly
    2 	Agree
    3 	Neither agree nor disagree
    4 	Disagree
    5 	Disagree strongly
    7 	Refusal*
    8 	Don't know*
    9 	No answer*
     "sofwrk" 	Society fair when hard-working people earn more than others
    1 	Agree strongly
    2 	Agree
    3 	Neither agree nor disagree
    4 	Disagree
    5 	Disagree strongly
    7 	Refusal*
    8 	Don't know*
    9 	No answer*
    """
    # TODO: Add remain/leave for all countries?
    actions_dict = {
        #'brexit' : ["vteumbgb"],
        'immigration': ["imbgeco"], # Immigration bad or good for the country's economy
        'lgbt_adopt' : ["hmsacld"], # Gay men and lesbians should have the same rights to adopt children as straight couples
        'lgbt_freedom' : ["freehms"], # Gay men and lesbians should be free to live life as they wish
    }
    """
    "vteumbgb" 	Voting intention if referendum was held tomorrow
    1 Remain
    0 Leave
    all other values Refusal
    """
    """
    "imbgeco" 	Immigration bad or good for country's economy
    0	Bad for the economy
    1	1
    2	2
    3	3
    4	4
    5	5
    6	6
    7	7
    8	8
    9	9
    10	Good for the economy
    77	Refusal*
    88	Don't know*
    99	No answer*
    
    "freehms"
    1 	Agree strongly
    2 	Agree
    3 	Neither agree nor disagree
    4 	Disagree
    5 	Disagree strongly
    7 	Refusal*
    8 	Don't know*
    9 	No answer*
    
    "hmsacld"
    1 	Agree strongly
    2 	Agree
    3 	Neither agree nor disagree
    4 	Disagree
    5 	Disagree strongly
    7 	Refusal*
    8 	Don't know*
    9 	No answer*
    """

    dicts = [values_dict, principle_dict, actions_dict]
    all_interested_cols = []
    for dict in dicts:
        for _, item_list in dict.items():
                all_interested_cols.extend(item_list)
    df = df.dropna(subset=all_interested_cols)

    # Remove irrelevant answers (refusal/don't know/no answer)
    #   count the number of irrelevant answers per country, to report back.
    for key, item_list in values_dict.items():
        for item in item_list:
            incorrect_value_responses = [7, 8, 9]
            df = df.loc[~df[item].isin(incorrect_value_responses)]
    for key, item_list in principle_dict.items():
        for item in item_list:
            incorrect_principle_responses = [7, 8, 9]
            df = df.loc[~df[item].isin(incorrect_principle_responses)]
    for key, item_list in actions_dict.items():
        for item in item_list:
            if item == "imbgeco":
                incorrect_action_responses = [77, 88, 99]
                df = df.loc[~df[item].isin(incorrect_action_responses)]
                df[item] = df[item] + 1
            elif item == "freehms" or item == "hmsacld":
                incorrect_action_responses = [7,8,9]
                df = df.loc[~df[item].isin(incorrect_action_responses)]
            elif item == "vteumbgb":
                incorrect_action_responses = [33,44,55,65]
                df = df.loc[~df[item].isin(incorrect_action_responses)]

    country_col_name = 'cntry'
    # NOTE: FOR UK DATA ONLY - because the UK value_systems has agents as citizens.
    #country_col_name = 'idno'
    #if country_col_name == 'idno':
    #    # remove all rows where country != UK
    #    df = df.loc[df['cntry'] == 'GB']
    #    print("verify DF, df size: ", df.shape)

    #value_preferences = process_all_country_values(df, country_col_name, values_dict, _, False)
    value_preferences = process_all_country_values(df, country_col_name, values_dict, higher_order_values_index_list, True)
    action_judgements = process_all_country_actions(df, country_col_name, value_preferences, actions_dict)
    # Because principle preferences are just preferences of each principle over every other principle, use the same func.
    principle_preferences = process_all_country_values(df, country_col_name, principle_dict, _, False)
    now = dt.now().isoformat()
    principles_fn = now+"_ess_principles.csv"
    # Principles:
    with open(principles_fn, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Country", "Principle"])
        for country, value in principle_preferences.items():
            writer.writerow([country, value])

        # Values + Preferences + Action Judgements
        #value_names = list(values_dict.keys())
        value_names = list(higher_order_values_dict.keys())
        action_names = list(actions_dict.keys())

        wide_fn = now + "_ess_value_system.csv"
        with open(wide_fn, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            header = ["country"]
            # P__vi__vj columns
            for vi in value_names:
                for vj in value_names:
                    header.append(f"P__{vi}__{vj}")
            # VA__v__a columns
            for v in value_names:
                for a in action_names:
                    header.append(f"VA__{v}__{a}")
            writer.writerow(header)
            # Rows
            for country in df[country_col_name].unique():
                P = np.array(value_preferences[country], dtype=float)
                VA = np.array(action_judgements[country], dtype=float)

                row = [country]
                for i in range(len(value_names)):
                    for j in range(len(value_names)):
                        row.append(float(P[i, j]))

                for i in range(len(value_names)):
                    for k in range(len(action_names)):
                        row.append(float(VA[i, k]))

                writer.writerow(row)