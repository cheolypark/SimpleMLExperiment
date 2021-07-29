import pandas as pd
from src.util.dataframe import load_file, save_file

import os
import sys
from os import listdir
from os.path import isfile, join

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(file_path, "..", "..", ".."))
sys.path.insert(0,os.path.join(file_path, "..", ".."))
sys.path.insert(0,os.path.join(file_path, ".."))


#==============================================================================
# 1 Load all csv files and combine as one dataframe
#==============================================================================
data_path = '../../../test/data/ai_tutor/raw/'

data_directories = [f for f in listdir(data_path)]

def get_dataframe (dir, file, number):
    df = pd.read_csv(os.path.join(dir, file))

    if 'user_id' in df.columns:
        # df['user_id'] = df['user_id'].str.replace('user',f'{number}_user')
        df['user_id'] = df['user_id'].astype(str) + '_' + number

    if 'current_problem_id' in df.columns:
        df = df.rename(columns={"current_problem_id": "user_current_lecture"})

    if 'current_problem' in df.columns:
        df = df.rename(columns={"current_problem": "user_current_lecture"})

    return df

current_problem_table = []
flowchart_syntax_element_table = []
lecture_table = []
rel__between__current_problem__and__syntax_element = []
rel__between__lecture__and__syntax_element = []
rel__current_problem__and__suggestion = []
suggestion_table = []
user_completed_lecture_table = []
user_table = []

number = 1
for d in data_directories:
    dir = data_path + d
    data_files = [f for f in listdir(dir) if isfile(join(dir, f))]
    number = d
    current_problem_table.append(get_dataframe(dir, 'current_problem_table.csv', number))
    flowchart_syntax_element_table.append(get_dataframe(dir, 'flowchart_syntax_element_table.csv', number))
    lecture_table.append(get_dataframe(dir, 'lecture_table.csv', number))
    rel__between__current_problem__and__syntax_element.append(get_dataframe(dir, 'rel__between__current_problem__and__syntax_element.csv', number))
    rel__between__lecture__and__syntax_element.append(get_dataframe(dir, 'rel__between__lecture__and__syntax_element.csv', number))
    rel__current_problem__and__suggestion.append(get_dataframe(dir, 'rel__current_problem__and__suggestion.csv', number))
    suggestion_table.append(get_dataframe(dir, 'suggestion_table.csv', number))
    user_completed_lecture_table.append(get_dataframe(dir, 'user_completed_lecture_table.csv', number))
    user_table.append(get_dataframe(dir, 'user_table.csv', number))

    # number += 1

current_problem_table = pd.concat(current_problem_table)
flowchart_syntax_element_table = pd.concat(flowchart_syntax_element_table)
lecture_table = pd.concat(lecture_table)
rel__between__current_problem__and__syntax_element = pd.concat(rel__between__current_problem__and__syntax_element)
rel__between__lecture__and__syntax_element = pd.concat(rel__between__lecture__and__syntax_element)
rel__current_problem__and__suggestion = pd.concat(rel__current_problem__and__suggestion)
suggestion_table = pd.concat(suggestion_table)
user_completed_lecture_table = pd.concat(user_completed_lecture_table)
user_table = pd.concat(user_table)

current_problem_table = current_problem_table.reset_index(drop=True)
rel__current_problem__and__suggestion = rel__current_problem__and__suggestion.reset_index(drop=True)


#==============================================================================
# 2 Join dataframes
#==============================================================================
# df_user_problem = pd.concat([current_problem_table, rel__current_problem__and__suggestion], axis=1, ignore_index=True, join="inner")
df_user_problem = pd.concat([current_problem_table, rel__current_problem__and__suggestion], axis=1)

df_user_problem = df_user_problem.iloc[:,~df_user_problem.columns.duplicated()]

print(df_user_problem)

df_user_problem = pd.merge(df_user_problem,
                           rel__between__current_problem__and__syntax_element,
                           how='left', on=['time_stamp', 'user_id', 'user_current_lecture'])

# drop missing values
# contaminated_region = regional_individual_patient[['latitude', 'longitude']].dropna()

def region_17_preprocessing(cases_17_region, window=8):
    # Data columns
    #   prev_confirmed: t-1 of confirmed
    #   new_confirmed: confirmed - prev_confirmed
    #   pop_confirmed: confirmed/population
    #   pop_new_confirmed: new_confirmed/population
    #   pop_new_confirmed_t_1: t-1 of pop_new_confirmed
    #   pop_new_confirmed_t_2: t-2 of pop_new_confirmed
    #   diff_pop_new_confirmed: pop_new_confirmed - pop_new_confirmed_t_1
    #   diff_pop_new_confirmed_t_1: t-1 of diff_pop_new_confirmed
    #   diff_pop_new_confirmed_t_2: t-2 of diff_pop_new_confirmed

    # 1 Group by province
    provinces = cases_17_region.groupby("province")
    provinces_df = []
    for s, s_df in provinces:
        # 2 Add previous confirmed
        s_df["prev_confirmed"] = s_df["confirmed"].shift(1)

        # 3 Add new confirmed
        # s_df["new_confirmed"] = s_df["confirmed"] - s_df["prev_confirmed"]
        # s_df["prev_new_confirmed"] = s_df["new_confirmed"].shift(1)

        # 4 Apply population for the case data
        # s_df["pop_confirmed"] = s_df["confirmed"]/s_df["population"]
        # s_df["pop_new_confirmed"] = s_df["new_confirmed"]/s_df["population"]
        #
        # # 5 Add t-1 of pop_new_confirmed
        # for t in range(window):
        #     s_df[f"pop_new_confirmed_t_{t+1}"] = s_df["pop_new_confirmed"].shift(t+1)

        # 6 difference of pop_new_confirmed
        # s_df["diff_pop_new_confirmed"] = s_df["pop_new_confirmed"] - s_df["pop_new_confirmed_t_1"]
        #
        # # 7 Add t-1 of diff_pop_new_confirmed
        # for t in range(window):
        #     s_df[f"diff_pop_new_confirmed_t_{t+1}"] = s_df["diff_pop_new_confirmed"].shift(t+1)

        # 8 Add ratio between confirmed and prev_confirmed
        s_df["confirmed_with_one"] = s_df["confirmed"].apply(lambda x: 1 if x == 0 else x)
        s_df["prev_confirmed_with_one"] = s_df["confirmed_with_one"].shift(1)

        s_df["ratio_confirmed"] = s_df["confirmed_with_one"]/s_df["prev_confirmed_with_one"]

        # 9 Add t-1 of ratio_confirmed
        for t in range(window):
            s_df[f"ratio_confirmed_t_{t+1}"] = s_df["ratio_confirmed"].shift(t+1)

        # 10 Add policy information to each province
        #  - 1, Korea,Alert,Infectious Disease Alert Level,Level 1 (Blue),1/3/2020,1/19/2020
        #  - 2, Korea,Alert,Infectious Disease Alert Level,Level 2 (Yellow),1/20/2020,1/27/2020
        #  - 3, Korea,Alert,Infectious Disease Alert Level,Level 3 (Orange),1/28/2020,2/22/2020
        #  - 4, Korea,Alert,Infectious Disease Alert Level,Level 4 (Red),2/23/2020,
        s_df["alert_level"] = 1
        s_df.index = pd.to_datetime(s_df['date'])
        s_df.loc['1/20/2020':'1/27/2020', 'alert_level'] = 2
        s_df.loc['1/28/2020':'2/22/2020', 'alert_level'] = 3
        s_df.loc['2/23/2020':'6/30/2020', 'alert_level'] = 4

        # 11 Add t-1 of policy
        for t in range(window):
            s_df[f"alert_level_t_{t+1}"] = s_df["alert_level"].shift(t+1)

        # 12 Add data to province
        provinces_df.append(s_df)

    # Drop rows containing null
    df = pd.concat(provinces_df)

    # Warning: When dropping data for nan, rows or cases are removed, so be careful to set the window size
    df = df.dropna()
    df = df.reset_index(drop=True)
    # df = df.drop(columns=['index'])

    return df

look_back_for_data = 8

cases_17_region = region_17_preprocessing(cases_17_region, look_back_for_data)
save_file(cases_17_region, "../../test/data/covid/input/cases_17_region.csv")

print(cases_17_region)
