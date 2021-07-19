from bayes_opt import BayesianOptimization
from src.lib.manager_machine_learning import ManagerMachineLearning

#================================================================
# Reference:
# https://github.com/fmfn/BayesianOptimization
#================================================================


class ManagerOptimization(ManagerMachineLearning):
    def __init__(self):
        super().__init__()
        self.ob_results = []
        self.set_iter()
        self.set_init_points()
        self.pbounds = None

    def set_iter(self, n_iter=5):
        self.n_iter = n_iter

    def set_init_points(self, init_points=5):
        self.init_points = init_points

    def set_pbounds(self, pbounds):
        # Bounded region of parameter space
        # e.g.,) pbounds = {'x': (2, 4), 'y': (-3, 3)}
        self.pbounds = pbounds

    def run_ml_multiple_times(self):
        if self.pbounds is not None:
            return self.perform_optimization()
        else:
            return super().run_ml_multiple_times()

    def perform_optimization(self):
        try:
            optimizer = BayesianOptimization(f=self.ml_black_box, pbounds=self.pbounds, random_state=1)
            optimizer.maximize(n_iter=self.n_iter, init_points=self.init_points)
        except:
            print("!!! An exception occurred for Bayesian Optimization [May be a disconnected model] !!!")

        print(optimizer.max)
        self.ob_results.append(optimizer.max)

        return optimizer

    def ml_black_box(self, **arg):
        #**********************************************************************************
        # Note that optimization works for only one ML algorithm, please set one ML algorithm in the code
        #**********************************************************************************

        # 1 Convert float to integer
        arg = {k : int(v) for k, v in arg.items() if k == 'n_estimators' or k == 'neurons' or k == 'look_back_for_model'}
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('hyper:', arg)

        # 2 Reset hyper parameters for an ML algorithm
        self.init_ml_algorithms(hyper=arg)

        # 3 Perform ML
        avg_score = super().run_ml_multiple_times()
        print('ml_black_box avg_score:', avg_score)

        # 4 Return score
        return -avg_score[0]

