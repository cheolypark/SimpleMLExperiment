import pandas as pd
import warnings
from src.util.dataframe import load_file, save_file

global_air_traffic = load_file('../../test/data/covid/Global_airport_traffic_data.csv')
mobility = load_file('../../test/data/covid/2020_KR_Region_Mobility_Report.csv')
policy = load_file('../../test/data/covid/Policy_Cleaned.csv')
cases_17_region = load_file('../../test/data/covid/TimeProvince.csv')
car_movement = load_file('../../test/data/covid/Movement_2019_2020_7.csv')
region_info = load_file('../../test/data/covid/Region.csv')
individual_patient = load_file('../../test/data/covid/PatientInfo.csv')
region_17_info = load_file('../../test/data/covid/Region17Info.csv')
flights_china = load_file('../../test/data/covid/flightlist_China.csv')

regional_individual_patient = pd.merge(individual_patient[['patient_id', 'confirmed_date', 'sex', 'age', 'province', 'city']],
                                       region_info[['province', 'city', 'latitude', 'longitude']], how='left', on=['province', 'city'])

cases_17_region = pd.merge(cases_17_region, region_17_info, how='left', on=['province'])

# drop missing values
contaminated_region = regional_individual_patient[['latitude', 'longitude']].dropna()


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
