import xgboost
from collections import OrderedDict


class XGBoost():
    def __init__(self):
        self.col_cat = []

        self.colsample_bytree=0.4,
        self.gamma=0,
        self.learning_rate=0.07,
        self.max_depth=3,
        self.min_child_weight=1.5,
        self.n_estimators=10000,
        self.reg_alpha=0.75,
        self.reg_lambda=0.45,
        self.subsample=0.6,
        self.seed=42

    def fit(self, X_train, y_train):
        # self.model = xgboost.XGBRegressor(colsample_bytree=self.colsample_bytree ,
        #                                   gamma=self.gamma,
        #                                   learning_rate=self.learning_rate,
        #                                   max_depth=self.max_depth,
        #                                   min_child_weight=self.min_child_weight,
        #                                   n_estimators=self.n_estimators,
        #                                   reg_alpha=self.reg_alpha,
        #                                   reg_lambda=self.reg_lambda,
        #                                   subsample=self.subsample,
        #                                   seed=self.seed)

        self.model = xgboost.XGBRegressor()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def feature_importance(self):
        tmp = OrderedDict(sorted(self.model.booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
        print(tmp)