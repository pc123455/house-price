#!/usr/bin/python
# -*- coding: utf-8 -*-

import lightgbm as lgb
from sklearn import linear_model
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation


model_lgb = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              metric = 'l2',
                              max_bin = 128,
                              num_trees = 5000,
                              num_leaves = 1024,
                              learning_rate = 0.005,
                              feature_fraction = 0.8,
                              bagging_fraction = 0.7,
                              bagging_freq = 1,
                              verbose = 0,
                              num_threads = 2,
                              lambda_l1 = 0.5,
                              min_data_in_leaf = 10,
                              bagging_seed =3,
                              early_stopping_round = 100,
                              min_data_per_group = 20)

model_lasso = linear_model.Lasso(alpha=0.0001, copy_X=True)
model_ridge = linear_model.Ridge(alpha=1, copy_X=True)


class EnsenbleModel:

    def __init__(self, **kwargs):
        model_ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('ridge', linear_model.Ridge(alpha=1200, copy_X=True))])
        model_lasso = linear_model.Lasso(alpha=0.0001, copy_X=True)
        model_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                                      objective='regression',
                                      metric='l2',
                                      max_bin=128,
                                      num_trees=5000,
                                      num_leaves=1024,
                                      learning_rate=0.05,
                                      feature_fraction=0.8,
                                      bagging_fraction=0.7,
                                      bagging_freq=1,
                                      verbose=0,
                                      num_threads=2,
                                      lambda_l1=0.5,
                                      min_data_in_leaf=15,
                                      bagging_seed=3,
                                      early_stopping_round=100,
                                      min_data_per_group=20)
        self.models = {'ridge': model_ridge, 'lasso': model_lasso, 'lgb': model_lgb}

    def fit(self, X, y, **kwargs):
        for k, m in self.models.items():
            if k == 'lgb':
                train_X, val_X = cross_validation.train_test_split(X, test_size = 0.3, random_state = 0)
                train_y, val_y = cross_validation.train_test_split(y, test_size = 0.3, random_state = 0)
                m.fit(train_X, train_y, eval_set = (val_X, val_y))
            else:
                m.fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, m in enumerate(self.models.values()):
            predictions[:, i] = m.predict(X)
        return np.mean(predictions, axis = 1)

    def get_params(self, deep = False):
        return { 'alpha': 1 }