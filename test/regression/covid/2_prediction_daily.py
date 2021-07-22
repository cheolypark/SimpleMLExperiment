import sys
import os
import pprint
from test.covid19_data_sets import look_back_for_data, cases_17_region

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(file_path, "..", "..", ".."))
sys.path.insert(0,os.path.join(file_path, "..", ".."))
sys.path.insert(0,os.path.join(file_path, ".."))

pp = pprint.PrettyPrinter(indent=1)
pp.pprint(sys.path)

import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from src.util.dataframe import save_file
from test.regression.covid.config import Config


class Prediction(Config):
    def __init__(self):
        super().__init__()
        self.cf['Learned_Model_Path'] = '../../data/covid/learned_model/'
        self.cf['Save_Results'] = True

        # 1 Load ML models
        super().load_models(self.cf['Learned_Model_Path'])

    def run_experiment_for_a_region(self, MAEs, test_province='Busan'):
        # 1 Select data
        super().select_data_in_row_selection_mode(test_province)

        # 2 Prepare the experiment dataset
        df = cases_17_region.where(cases_17_region['province'].isin(self.test_columns['province'])).dropna(subset=['province'])
        df = df.reset_index().drop(columns=['index'])
        df['predicted_confirmed'] = None

        # 3 Perform the confirmed case prediction experiment
        output_columns = []
        for ml in self.algorithms:
            name = type(ml).__name__
            predicted_confirmed = f'predicted_confirmed_{name}'
            output_columns.append(predicted_confirmed)

            # Get data
            test_X = self.X_test
            if name == 'LSTMRegressor':
                test_X = self.X_win_test

                #***************************************************************************************************
                # Warning: For the LSTM test, the below code removing the first a few data is applied.
                # This will influence the accuracy score for other MLs, so MAEs previously derived should be checked
                # for fair comparison
                df = df[(self.cf['Look_Back_For_Model'] + 1):]
                #***************************************************************************************************

            # 4 Perform prediction
            y_predict = ml.predict(test_X)
            y_predict = y_predict.ravel()

            # 5 Add the predicted ratio
            df['ratio_predicted'] = y_predict

            # 6 Convert the predicted ratio to the confirmed and add them into the dataset
            df[predicted_confirmed] = df['prev_confirmed_with_one'] * df['ratio_predicted']

            # 7 For the case of confirmed == 0, predicted_confirmed also becomes zero.
            # This prediction starts from a case when a least one confirmed occurs.
            df[predicted_confirmed] = np.where(df['confirmed'] == 0, df['confirmed'], df[predicted_confirmed])

            # 8 Measure accuracy performance
            mae = mean_absolute_error(df['confirmed'], df[predicted_confirmed])
            print(f'{name} \t', mae)
            MAEs[predicted_confirmed] = mae

        print(f'{output_columns}: Prediction was finished!')

        if self.cf['Save_Results'] is True:
             # 9 Save the predicted results
            csv_name = datetime.now().strftime('ml[' + test_province + "]_%m_%d_%Y-%H_%M_%S")
            result_excel = df[['date', 'province', 'confirmed'] + output_columns]
            save_file(result_excel, f'../../test/data/covid/output_prediction/{csv_name}.csv', index=False)

    def run(self):
        # 1 Perform predictions
        province_MAEs = {}
        ts = time.time()

        index = 1
        for test_province in self.all_province:
            MAEs = {}
            self.run_experiment_for_a_region(MAEs, test_province)
            province_MAEs[test_province] = MAEs

            print('====================================================================>', index, '/', len(self.all_province))
            index += 1

        # 2 Make Average MAEs
        df_maes = pd.DataFrame(province_MAEs)
        df_maes['Average MAEs'] = df_maes.mean(axis=1)

        # 3 Save MAEs
        if self.cf['Save_Results'] is True:
            csv_name = datetime.now().strftime('MAEs' + "_%m_%d_%Y-%H_%M_%S")
            save_file(df_maes, f'../../test/data/covid/output_prediction/{csv_name}.csv', index=False)

        print('<=== Average MAEs ===========================>')
        print(df_maes['Average MAEs'])
        print("== Mahcine learning end: Time {} mins ".format((time.time()-ts)/60))
        print('<============================================>')

        return df_maes['Average MAEs']


if __name__ == '__main__':
    experiment = Prediction()
    # - Test 17 regions
    experiment.run()

    # - Test only one region
    # MAEs = {}
    # experiment.run_experiment_for_a_region(MAEs, test_province='Busan')
