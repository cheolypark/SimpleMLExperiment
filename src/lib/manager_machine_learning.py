from src.lib.manager_ml_algorithms import ManagerMLAlgorithms
import pandas as pd
import datetime
import time
from src.util.dataframe import load_file, save_file

#====================================================================
# In row_selection_mode, test and training data are selected by a certain row value in a column
# e.g.,)
#   X     |       Y
#   a     |       1
#   a     |       2
#   b     |       3
#   c     |       4
#
#   Data associated with 'a' will be used for testing
#   Remaining data will be used for training


class ManagerMachineLearning(ManagerMLAlgorithms):
    def __init__(self):
        super().__init__()

    def prepare_data(self, dataset):
        # Set the main dataset
        self.df = dataset

        # Take only necessary data from the original data
        # self.cf['Columns'], self.cf['Target']
        self.all_columns = self.cf['Condition_Columns'] + self.cf['X_Columns'] + [self.cf['Target']]
        self.df = self.df[self.all_columns]

    def set_ml_mode(self, mode='Row_Selection_Mode', column_for_row_selection='province'):
        self.cf['ML_Mode'] = mode

        if mode == 'Row_Selection_Mode':
            self.column_for_row_selection = column_for_row_selection
            self.all_row_values = self.df[self.column_for_row_selection].unique()
            # self.all_row_values = ['Busan', 'Seoul',  'Incheon']

    #================================================================================
    # For "row_selection_mode"
    #================================================================================

    def select_data_in_row_selection_mode(self, row_value_for_test='Busan'):
        # e.g.,)
        #   X     |       Y
        #   a     |       1
        #   a     |       2
        #   b     |       3
        #   c     |       4
        # columns_for_row = 'X'
        # row_value_for_test = 'a'

        self.row_values_for_training = [x for x in self.all_row_values if x != row_value_for_test]
        self.train_columns = {self.column_for_row_selection: self.row_values_for_training}
        self.test_columns = {self.column_for_row_selection: [row_value_for_test]}

        self.set_train_test_data_by_rows(self.df, self.cf['X_Columns'], self.cf['Target'], self.train_columns, self.test_columns)

    def run_ml_in_row_selection_mode_for_each(self, test_province='Busan'):
        # 1 Select data
        self.select_data_in_row_selection_mode(test_province)

        # 2 Perform ML using the selected data
        result_scores = self.perform_ML()

        return result_scores

    def run_ml_in_row_selection_mode(self):
        ts = time.time()

        total_scores = {}
        index = 1
        for row_value_for_test in self.all_row_values:
            total_scores[row_value_for_test] = self.run_ml_in_row_selection_mode_for_each(row_value_for_test)
            print('====================================================================>', index, '/', len(self.all_row_values))
            index += 1

        # Make Average MAEs
        df_scores = pd.DataFrame(total_scores)
        df_scores['Average Scores'] = df_scores.mean(axis=1)

        # Save MAEs
        if self.cf['Save_Results']:
            csv_name = datetime.now().strftime('MAEs' + "_%m_%d_%Y-%H_%M_%S")
            save_file(df_scores, f'../../data/output_prediction/{csv_name}.csv', index=False)

        print(f'=== Score of {self.cf["Metrics"]} === ')
        print(df_scores['Average Scores'])
        print("Time: {:.2f} mins".format((time.time()-ts)/60))
        print('==========================================================>')

        return df_scores['Average Scores']

    def run_ml(self):
        if self.cf['ML_Mode'] == 'Row_Selection_Mode':
            return self.run_ml_in_row_selection_mode()
        else:
            pass

    def run_ml_multiple_times(self):
        ex_scores = None

        for ex in range(self.cf['Experiment_Times']):
            scores = self.run_ml()
            if ex_scores is None:
                ex_scores = scores
            else:
                ex_scores = pd.concat([ex_scores, scores], axis=1)

        # Get average

        print('=== Overall results ===')
        ex_means = ex_scores.mean(axis=1)
        ex_stds = ex_scores.std(axis=1)

        print('MEAN: ', ex_means)
        print('STD: ',ex_stds)
        print('================================================>')

        return ex_means


