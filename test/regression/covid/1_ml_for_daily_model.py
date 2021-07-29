import sys
import os
import pprint

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(file_path, "..", "..", ".."))
sys.path.insert(0,os.path.join(file_path, "..", ".."))
sys.path.insert(0,os.path.join(file_path, ".."))

pp = pprint.PrettyPrinter(indent=1)
pp.pprint(sys.path)

from tensorflow.keras.optimizers import Nadam
from test.regression.covid.config import Config


class MachineLearning(Config):
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
        self.cf['Learned_Model_Path'] = '../../test/data/covid/learned_model/'

        #========================================================
        # Set Deep learning config
        #========================================================
        self.cf['DL_TYPE'] = 'custom'
        self.cf['Epochs'] = 1500
        self.cf['Batch_Size'] = 256
        self.cf['Look_Back_For_Model'] = 4
        self.cf['Learning_Rate'] = 1e-5
        self.cf['Neurons'] = 64 * 1
        self.cf['Optimizer'] = Nadam(lr=self.cf['Learning_Rate'])
        self.cf['Save_Dir'] = 'saved_models'
        # self.cf['Optimizer'] = 'adam'

        self.cf['Layers'] = []

        #========================================================
        # Set parameter optimizer configs
        #========================================================
        # self.init_ml_algorithms(['LR'])
        # self.init_ml_algorithms(['DLR'])
        # self.init_ml_algorithms(['TRS'])
        self.init_ml_algorithms(['LSTM', 'DLR'])
        # self.init_ml_algorithms(['RFR'])
        # self.init_ml_algorithms(['LR', 'DTR'])
        # self.init_ml_algorithms(['LR', 'GPR', 'MLP', 'SVR', 'RFR', 'GBR'])
        # self.init_ml_algorithms(['RFR', 'GBR'])

        #========================================================
        # Set parameter optimizer configs
        #========================================================
        # self.set_pbounds({'n_estimators': (10, 2000)})
        # self.set_pbounds({'neurons': (20, 5000)})

    def run(self):
        self.run_ml_multiple_times()


experiment = MachineLearning()
experiment.run()

