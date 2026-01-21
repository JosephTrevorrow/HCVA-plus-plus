"""
In this file we read the raw data from the ESS and we process
the data according to the section 6 of the article.

Religious maps to Traditionalist (unintentionally so)
Non-Religious maps to Hedonist (unintentionally so)
div is the mapping to the single action (for/against basic income scheme)
'adp' is mentioned here but not used in the single action example. Here it is kept as a placeholder should two actions be required.
"""

"""
This file has been adapted from work by Lera-Leri et al.. You can find their repository here: https://github.com/RogerXLera/ValueSystemsAggregation
"""

import pandas as pd
import csv
import copy
import numpy as np

def process_action(element):
    category_dic = {
        4: 1,
        3: 2,
        2: 3,
        1: 4
    }
    try:
        # against the scheme 
        div_value = element['basinc']
        if div_value > 4:
            raise BaseException
        div_support = -(div_value - 2.5) / 1.5
        # for the scheme
        adp_value = category_dic[element['basinc']]
        adp_support = -(adp_value - 2.5) / 1.5
        return adp_support, div_support
    except BaseException:
        return None, None

def process_principle(dataframe, caseno):
    """
    In this function we count the number of religious or non-religious
    participants per country according to the proceeding Serramià et al. describe
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
           caseno (number of case per country)
    Return: tuple (tuple with three values: religious: True if religious
                                            adp: float from -1, 1
                                            div: float from -1, 1)
    """
    # religious or not v6 in EVS2017
    dataframe_row = dataframe[dataframe['idno'] == caseno].iloc[0]
    # smdfslv here is the variable that represents the egalitarianism, but this has been dropped
    egalitarianism = dataframe_row['smdfslv']

    try:
        if egalitarianism == 5 or egalitarianism == 4:
            egalitarian = False
        elif egalitarianism == 2 or egalitarianism == 1:
            egalitarian = True
        else:
            egalitarian = None
    except BaseException:
        egalitarian = None

    if egalitarian is None:
        return None
    else:
        return (egalitarian)

def process_all_country(ess_df, country_col_name, values_dict):
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
        for value in values_list:
            compare_list = values_list.remove(value)
            for compare_val in compare_list:
                diff = value - compare_val
                
                

            
    return
    


def process_country(dataframe, country, values_dict):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the ESS)
           country (code of the european country, e.g. ES for Spain)
    Return: dict with # of religious and non-religious citizens,
            a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    """
    df = dataframe[dataframe['cntry'] == country]
    n_row = df.shape[0]

    # setting counters to compute the mean of each judgement value:
    # a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    # Religous or not are our values
    n_religious = 0
    n_nonreligious = 0
    n_rel_adp = 0
    n_nonrel_adp = 0
    n_rel_div = 0
    n_nonrel_div = 0
    sum_a_adp_rel = 0
    sum_a_adp_nonrel = 0
    sum_a_div_rel = 0
    sum_a_div_nonrel = 0

    for i in range(0, n_row):
        caseno = df.iloc[i]['idno']
        tuple_ = process_participant(df, caseno, values_dict)  # information of the case
        if tuple_:
            if tuple_[0]:  # True for religious citizens
                n_religious += 1
                # ignore missing data
                if tuple_[1] is not None:  # adopt judgement
                    n_rel_adp += 1
                    sum_a_adp_rel += tuple_[1]
                if tuple_[2] is not None:  # divorce judgement
                    n_rel_div += 1
                    sum_a_div_rel += tuple_[2]
            else:  # non-religious citizens
                n_nonreligious += 1
                if tuple_[1] is not None:  # adopt judgement
                    n_nonrel_adp += 1
                    sum_a_adp_nonrel += tuple_[1]
                if tuple_[2] is not None:  # divorce judgement
                    n_nonrel_div += 1
                    sum_a_div_nonrel += tuple_[2]
        else:
            continue

    # Normalise the preferences and add to new vars
    normalised_religious = n_religious / (n_religious+n_nonreligious)
    normalised_nonreligious = n_nonreligious / (n_religious+n_nonreligious)
    print("country", country)
    return {
        'rel_sum': n_religious,
        'nonrel_sum': n_nonreligious,
        'rel' : normalised_religious,
        'nonrel' : normalised_nonreligious,
        'a_div_rel': sum_a_div_rel / n_rel_div,
        'a_div_nonrel': sum_a_div_nonrel / n_nonrel_div
    }

def principle_process_country(dataframe, country):
    """
    Process information for each country
    INPUT: dataframe (pandas dataframe with the results of the EVS 2017)
           country (code of the european country, e.g. ES for Spain)
    Return: dict with # of religious and non-religious citizens,
            a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    """
    df = dataframe[dataframe['cntry'] == country]
    n_row = df.shape[0]
    # setting counters to compute the mean of each judgement value:
    # a_rl(ad), a_pr(ad), a_rl(dv) and a_pr(dv)
    n_religious = 0
    n_nonreligious = 0

    for i in range(0, n_row):
        caseno = df.iloc[i]['idno']
        value = process_principle(df, caseno)  # information of the case
        if value:
            n_religious += 1
        else:  # non-religious citizens
            n_nonreligious += 1

    print("country", country)
    print('n_religious: ', n_religious)
    print('n_nonreligious: ', n_nonreligious)
    return {
        'rel': n_religious,
        'nonrel': n_nonreligious,
    }

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
        #'brexit' : ["vteumbgb"],
        'immigration': ["imbgeco"] # Immigration bad or good for country's economy
    }

    dicts = [values_dict, principle_dict, actions_dict]
    all_interested_cols = []
    for dict in dicts:
        for _, item_list in dict.items():
                all_interested_cols.extend(item_list)
    print(all_interested_cols)
    df = df.dropna(subset=all_interested_cols)
    print(df)

    country_col_name = 'cntry'
    process_all_country(df, country_col_name, values_dict)

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
