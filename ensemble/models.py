#!/usr/bin/python
# -*- coding: utf-8 -*-

import lightgbm as lgb
from sklearn import linear_model
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
import xgboost as xgb

class EnsenbleModel:

    def __init__(self, **kwargs):
        model_ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('ridge', linear_model.Ridge(alpha=1100, copy_X=True))])
        model_lasso = linear_model.Lasso(alpha=0.0001, copy_X=True)
        model_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                                      objective='regression',
                                      metric='l2',
                                      max_bin=32,
                                      num_trees=5000,
                                      num_leaves=4,
                                      max_depth=2,
                                      learning_rate=0.01,
                                      feature_fraction=0.5,
                                      bagging_fraction=0.6,
                                      bagging_freq=2,
                                      verbose=0,
                                      num_threads=2,
                                      lambda_l1=0.05,
                                      lambda_l2 = 1,
                                      min_data_in_leaf=10,
                                      bagging_seed=3,
                                      early_stopping_round=100,
                                      min_data_per_group=20)
        model_xgb = xgb.XGBRegressor(max_depth=2,
                                    learning_rate=0.01,
                                    n_estimators=4000,
                                    objective='reg:linear',
                                    nthread=-1,
                                    gamma=0.01,
                                    min_child_weight=2,
                                    max_delta_step=0,
                                    subsample=0.80,
                                    colsample_bytree=0.5,
                                    colsample_bylevel=0.5,
                                    reg_alpha=0.01,
                                    reg_lambda=0.5,
                                    scale_pos_weight=1,
                                    seed=1440,
                                    missing=None)
        model_linear_xgb = xgb.XGBRegressor(max_depth=3,
                                     booster='gblinear',
                                     learning_rate=0.8,
                                     n_estimators=6000,
                                     early_stopping_rounds = 100,
                                     # silent=True,
                                     objective='reg:linear',
                                     nthread=-1,
                                     gamma=0.01,
                                     min_child_weight=2,
                                     max_delta_step=0,
                                     subsample=0.70,
                                     colsample_bytree=0.6,
                                     colsample_bylevel=0.6,
                                     reg_alpha=0.02,
                                     reg_lambda=1,
                                     scale_pos_weight=1,
                                     seed=1440,
                                     missing=None)
        # self.models = {'ridge': model_ridge, 'lasso': model_lasso, 'lgb': model_lgb, 'xgb': model_linear_xgb}
        self.models = {'xgb': model_linear_xgb}

    def fit(self, X, y, **kwargs):
        for k, m in self.models.items():
            if k == 'lgb':
                train_X, val_X = cross_validation.train_test_split(X, test_size = 0.3, random_state = 0)
                train_y, val_y = cross_validation.train_test_split(y, test_size = 0.3, random_state = 0)
                m.fit(train_X, train_y, eval_set = (val_X, val_y))
            elif k == 'xgb':
                train_X, val_X = cross_validation.train_test_split(X, test_size=0.3, random_state=0)
                train_y, val_y = cross_validation.train_test_split(y, test_size=0.3, random_state=0)
                m.fit(train_X, train_y, eval_set=[(val_X, val_y)])
            else:
                m.fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, m in enumerate(self.models.values()):
            predictions[:, i] = m.predict(X)
        return np.mean(predictions, axis = 1)

    def get_params(self, deep = False):
        return { 'alpha': 1 }