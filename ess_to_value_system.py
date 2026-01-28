"""
This file has been adapted from work by Lera-Leri et al.. You can find their repository here: https://github.com/RogerXLera/ValueSystemsAggregation
"""

import pandas as pd
import csv
import copy
import numpy as np

def process_all_country_values(ess_df, country_col_name, values_dict):
    """
    Docstring for process_all_country
    
    :param ess_df: The dataframe containing all ESS data
    :param country_col_name: typically 'cntry', the col that will state which country each row belongs to
    :param values_dict: Dict of form str(value) : [question_id's that relate to each value]
    """
    # Find the list of all countries
    ess_country_list = ess_df[country_col_name].unique()

    # Iterate through each country and find their values
    country_values = {}
    for country in ess_country_list:
        country_df = ess_df.loc[ess_df[country_col_name] == country]
        temp_values = []
        # Iterate through and find all values, summing for each 
        #   question. temp_values will be in sameZ order as values_dict
        # Then find the central preference by finding mean of each of these.
        for _, values in values_dict.items():
            temp_questions = []
            for question in values:
                # Invert all the items such that higher scores represent greater importance (inverted = 1 = worst, 6 = best)
                country_df[question] = country_df[question].values[::-1]
                # Find mean score for every col
                mean_score = country_df[question].sum() / len(country_df)
                # temp_questions contains the mean score of every question for that specific value, in thesame order as the list for the values_dict
                temp_questions.append(copy.copy(mean_score))
            # temp_values contains the list of mean scores for each question for that value, in the same order as the values_dict
            temp_values.append(copy.copy(temp_questions))
        # country_values contains the list of mean scores for every question for the country
        country_values[country] = copy.copy(temp_values)
    # Subtract the mean score across all values (for a single country) from a value's raw score. This produces centrered values.
    # country_values = dict({country_name: [[value1_q1, value1_q2], [value2_q1, value2_q2], [..]]})
    for country, values in country_values.items():
        # Find the sum of all the values in the nested list (values), and compute the mean
        total = 0
        num_of_questions = 0
        for value_list in values:
            num_of_questions += len(value_list)
            total += sum(value_list)
        mean = total / num_of_questions

        # For every value in the nested list: values, subtract the mean from it, to find a
        #  centered value for that question. Then find the mean of that list
        for value_list in values:
            for value in value_list:
                value_list[value_list.index(value)] = value - mean
        temp_country_vals = np.array([np.mean(v) for v in country_values[country]])  
        country_values[country] = copy.copy(temp_country_vals) 

    print("VERIFICATION, FR VALUES: ", country_values['FR'])

    ## Now we have a list of centred country_values for every country, 
    # we find the preferences between each value and every other value, and store
    value_preferences = {}
    for country, values_list in country_values.items():
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
    print("---\n")
    print("Value Preferences:\n", value_preferences['AT'])

    # TODO: normalise the value preferences

    return value_preferences
    
def process_all_country_actions(ess_df, country_col_name, value_preferences, actions_dict, values_dict):
    """
    Given a country's value preferences, find their action value judgements
    INPUT: ess_df 
        country_col_name
        value_preferences: NxN np.array, comparing each value to every other value
        actions_dict: dict of actions in form str(action_name) : question_id
    """
    # Find the list of all countries
    ess_country_list = ess_df[country_col_name].unique()

    # First, iterate through each country and find their actions
    country_actions = {}
    for country in ess_country_list:
        country_df = ess_df.loc[ess_df[country_col_name] == country]
        temp_actions = []
        # Iterate through and find all values, summing for each 
        #   question. temp_values will be in sameZ order as values_dict
        # Then find the central preference by finding mean of each of these.
        for _, action_id in actions_dict.items():
            centred_score = 0.0
            # No need to invert actions, as score between 1-10, where 10 is GOOD, 1 is BAD
            # Find mean score for action (1 action, no need to find average of set)
            mean = country_df[action_id].mean()
            mean_score = mean.iloc[0]
            # Centre the scores between -1 and 1. action_id is always a list of size 1
            # immigration is between 1-10, where 10 is GOOD, 1 is BAD, others are between 1-5
            if action_id[0] == "imbgeco":
                # Because each action is between 1-10
                centred_score = (((mean_score-1) * 2)/9) - 1
            else:
                # Actions are between 1-5, where 1 = agree strongly, 
                # and 5 = disagree strongly. 
                centred_score = (((mean_score-1) * 2)/4) - 1
            temp_actions.append(copy.copy(centred_score))
        # country_values contains the averaged, centred action for each action
        country_actions[country] = copy.copy(temp_actions)
    # Second, convert country actions into action judgements considering their value preferences
    action_judgements = {}
    for country in ess_country_list:
        country_values = value_preferences[country]
        actions = country_actions[country]
        # Convert responses to action judgements for every value considering the value preferences
        temp_action_judgements = {}
        # for every cell in the top triangle of the matrix (as it is symmetric), check pref, add action judgement for that cell.
        #   Action judgement: A country that prefers value X over Y, supports the action Z amount.
        #   Mathematically: action judgement for value x>y = centred_action * value_preference_p>y
        # numpy arrays will iterate over each row
        value_names = list(values_dict.keys())

        """
        array([[[uni, uni], [uni, ben], [uni, tra]],
               [[ben, uni], [ben, ben], [ben, tra]],
               [[tra, uni], [tra, ben], [tra, tra]]])
        """
        # For every action
        actions_keys = list(actions_dict.keys())
        for action, x in zip(actions, range(0, len(actions_keys))):
            # For every value row (each row is every preference for one value)
            for value_pref_row, i in zip(country_values, range(0, len(country_values))):
                # for every value in the value row
                for value_pref, j in zip(value_pref_row, range(0, len(value_pref_row))):
                    # if the preference is positive (it prefers this value more than another)
                    # TODO: Now prefs are normalised, the preferences are between 0-1, so midpoint is 0.5!
                    pref_strength = 2*(value_pref-0.5)
                    to_save = action * pref_strength
                    if value_pref > 0.5:
                        temp_key = actions_keys[x] + "_" + value_names[i] + "_over_" + value_names[j]
                        temp_action_judgements[temp_key] = copy.copy(to_save)
                    else:
                        temp_key = actions_keys[x] + "_" + value_names[j] + "_over_" + value_names[i]
                        temp_action_judgements[temp_key] = copy.copy(to_save)
            action_judgements[country] = copy.deepcopy(temp_action_judgements)
    return action_judgements

def process_all_country_principles(ess_df, country_col_name, principles_dict):
    """
    
    """
    
    return

if __name__ == '__main__':
    df = pd.read_csv("ESS9-private/ESS9e03_2.csv")

    higher_order_values_dict = {
        "Self-Transcendence" : ["Universalism", "Benevolence", ], 
        "Conservation" : ["Tradition", "Conformity", "Security"], 
        "Self-Enhancement" : ["Power", "Achievement", "Hedonism"], 
        "Openness to Change" : ["Hedonism", "Stimulation", "Self-Direction"]
    }

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

    principle_dict = {
        "income_dist" : ["sofrdst"],  # Society fair when income and wealth is equally distributed
        "work_earnings" : ["sofrwrk"], # Society fair when hard-working people earn more than others
        "care_of_poor" : ["sofrpr"] # Society fair when takes care of poor and in need, regardless of what give back 
    }

    actions_dict = {
        #'brexit' : "vteumbgb",
        'immigration': ["imbgeco"] # Immigration bad or good for country's economy
    }
    """
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
    """

    dicts = [values_dict, principle_dict, actions_dict]
    all_interested_cols = []
    for dict in dicts:
        for _, item_list in dict.items():
                all_interested_cols.extend(item_list)
    print(all_interested_cols)
    df = df.dropna(subset=all_interested_cols)
    print(df)

    country_col_name = 'cntry'
    value_preferences = process_all_country_values(df, country_col_name, values_dict)
    process_all_country_actions(df, country_col_name, value_preferences, actions_dict, values_dict)



    # Old start:
    # we create a temp dictionary to store the data per country
    dictionary = {}
    for country in list(df['cntry'].unique()):
        temp_list = ['cntry', 'idno']
        temp_list.extend(all_interested_cols)
        print(temp_list)
        dict_ = process_country(
            df[temp_list], country, values_dict)
        dictionary.update({country: dict_})
    columns = ['country']
    for key in dictionary[country].keys():
        columns.append(key)
    csv_rows = [columns]
    for country in dictionary.keys():
        csv_rows2 = [country]
        for item in dictionary[country].keys():
            csv_rows2.append(dictionary[country][item])
        csv_rows.append(csv_rows2)
    
    # we store the data in a file
    #with open('principle_processed_data_ess.csv', 'w', newline='') as csvfile:

    with open('20-01-2026-agent-data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    # Old end
