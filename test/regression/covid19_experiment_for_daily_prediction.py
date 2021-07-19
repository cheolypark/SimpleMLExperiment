import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from src.util.dataframe import load_file, save_file
from tensorflow.keras.optimizers import Nadam
from test.regression.covid19_experiment import MLExperiment
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class Prediction(MLExperiment):
    def __init__(self):
        super().__init__()
        #========================================================
        # Set experiment configs
        #========================================================
        self.cf['Save_Results'] = False
        self.cf['Experiment_Times'] = 2
        self.cf['Verbose'] = 1
        # self.cf['Verbose'] = 0
        self.cf['Metrics'] = 'mean_absolute_error'

        #========================================================
        # Set Deep learning config
        #========================================================
        self.cf['DL_TYPE'] = 'custom'
        self.cf['Epochs'] = 1500
        # self.cf['Epochs'] = 1000
        self.cf['Batch_Size'] = 256
        self.cf['Look_Back_For_Model'] = 4
        self.cf['Learning_Rate'] = 1e-5
        self.cf['Save_Dir'] = 'saved_models'
        # self.cf['Optimizer'] = 'adam'
        self.cf['Optimizer'] = Nadam(lr=self.cf['Learning_Rate'])
        self.cf['Neurons'] = 64 * 1
        self.cf['Layers'] = []

        #========================================================
        # Set parameter optimizer configs
        #========================================================
        # self.init_ml_algorithms(['LR'])
        # self.init_ml_algorithms(['DLR'])
        self.init_ml_algorithms(['TRS'])
        # self.init_ml_algorithms(['LSTM'])
        # self.init_ml_algorithms(['RFR'])
        # self.init_ml_algorithms(['LR', 'DTR'])
        # self.init_ml_algorithms(['LR', 'GPR', 'MLP', 'SVR', 'RFR', 'GBR'])
        # self.init_ml_algorithms(['RFR', 'GBR'])

        #========================================================
        # Set parameter optimizer configs
        #========================================================
        # self.set_pbounds({'n_estimators': (10, 2000)})
        # self.set_pbounds({'neurons': (20, 5000)})

    def run_experiment_for_a_region(self, MAEs, test_province='Busan'):
        # 1 Build models
        super().build_models(test_province)

        # 2 Prepare the experiment dataset
        df = self.df.where(self.df['province'].isin(self.test_columns['province'])).dropna(subset=['province'])
        df = df.reset_index().drop(columns=['index'])
        df['predicted_confirmed'] = None

        # 3 Perform the confirmed case prediction experiment
        output_columns = []
        for ml in self.ml.regressors:
            name = type(ml).__name__
            predicted_confirmed = f'predicted_confirmed_{name}'
            output_columns.append(predicted_confirmed)

            # Get data
            test_X = self.ml.X_test
            if name == 'LSTMRegressor':
                test_X = self.ml.X_win_test

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

        if self.save_result:
             # 9 Save the predicted results
            csv_name = datetime.now().strftime('ml[' + test_province + "]_%m_%d_%Y-%H_%M_%S")
            result_excel = df[['date', 'province', 'confirmed'] + output_columns]
            save_file(result_excel, f'../../data/output_prediction/{csv_name}.csv', index=False)

    def run(self):
        self.run_ml_multiple_times()


experiment = Prediction()
experiment.run()

