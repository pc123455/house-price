#!/usr/bin/python
# -*- coding: utf-8 -*-

import lightgbm as lgb
from sklearn import linear_model
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.model_selection import KFold

class EnsembleModel:

    def __init__(self, **kwargs):
        # model_ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
        #               ('ridge', linear_model.Ridge(alpha=200, copy_X=True))])
        model_ridge = linear_model.Ridge(alpha=5, copy_X=True)
        model_lasso = linear_model.Lasso(alpha=0.0001, copy_X=True)
        model_ela = linear_model.ElasticNet(alpha=0.002, l1_ratio=0.0005)
        model_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                                      objective='regression',
                                      metric='l2_root',
                                      max_bin=32,
                                      num_trees=5000,
                                      num_leaves=5,
                                      max_depth=-1,
                                      learning_rate=0.01,
                                      feature_fraction=0.25,
                                      bagging_fraction=0.7,
                                      bagging_freq=1,
                                      verbose=0,
                                      num_threads=2,
                                      lambda_l1=0.005,
                                      lambda_l2 = 0.8,
                                      min_sum_hessian_in_leaf=1e-2,
                                      min_data_in_leaf=8,
                                      bagging_seed=3,
                                      early_stopping_round=100,
                                      min_data_per_group=8)
        model_xgb = xgb.XGBRegressor(max_depth=2,
                                    learning_rate=0.01,
                                    n_estimators=5000,
                                    objective='reg:linear',
                                    nthread=-1,
                                    gamma=0.01,
                                    min_child_weight=15,
                                    max_delta_step=0,
                                    subsample=0.30,
                                    colsample_bytree=0.5,
                                    colsample_bylevel=0.5,
                                    reg_alpha=0.01,
                                    reg_lambda=0.5,
                                    scale_pos_weight=1,
                                    seed=1440,
                                    missing=None)
        model_linear_xgb = xgb.XGBRegressor(max_depth=3,
                                     booster='gblinear',
                                     silent = 0,
                                     learning_rate=0.8,
                                     n_estimators=6000,
                                     early_stopping_rounds = 10,
                                     objective='reg:linear',
                                     nthread=-1,
                                     gamma=0.01,
                                     min_child_weight=10,
                                     max_delta_step=0,
                                     subsample=0.3,
                                     colsample_bytree=0.5,
                                     colsample_bylevel=0.5,
                                     reg_alpha=0.14,
                                     reg_lambda=1,
                                     scale_pos_weight=1,
                                     seed=1440,
                                     eval_metric = 'rmse',
                                     missing=None)
        # self.models = {'ridge': model_ridge, 'lasso': model_lasso, 'ela': model_ela, 'lgb': model_lgb, 'xgb_linear': model_linear_xgb}
        self.models = {'lgb': model_lgb}

    def fit(self, X, y, **kwargs):
        for k, m in self.models.items():
            if k == 'lgb':
                train_X, val_X = cross_validation.train_test_split(X, test_size = 0.3, random_state = 0)
                train_y, val_y = cross_validation.train_test_split(y, test_size = 0.3, random_state = 0)
                m.fit(train_X, train_y, eval_set = (val_X, val_y))
            elif k == 'xgb' or k == 'xgb_linear':
                train_X, val_X = cross_validation.train_test_split(X, test_size=0.3, random_state=0)
                train_y, val_y = cross_validation.train_test_split(y, test_size=0.3, random_state=0)
                m.fit(train_X, train_y, eval_set = [(val_X, val_y)])
            else:
                m.fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, m in enumerate(self.models.values()):
            predictions[:, i] = m.predict(X)
        return np.mean(predictions, axis = 1)

    def multi_predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, m in enumerate(self.models.values()):
            predictions[:, i] = m.predict(X)
        return predictions

    def get_params(self, deep = False):
        return { 'alpha': 1 }


class Stacking:

    def __init__(self, kfold = 5, **kwargs):
        self.kfold = kfold
        self.models = list()

    def fit(self, X, y, **kwargs):
        kf = KFold(n_splits=self.kfold, shuffle=False)
        predictions = None
        new_y = np.array([])

        #1st layer
        for train_idx, test_idx in kf.split(X):
            train_X = X[train_idx]
            train_y = y[train_idx]
            test_X = X[test_idx]
            test_y = y[test_idx]

            model = EnsembleModel()
            model.fit(train_X, train_y)
            self.models.append(model)
            predict = model.multi_predict(test_X)
            if predictions is None:
                predictions = predict
            else:
                predictions = np.concatenate((predictions, predict), axis = 0)
            new_y = np.concatenate((new_y, test_y), axis = 0)

        #2nd layer
        self.second_layer_model = linear_model.Ridge(alpha=0.01)
        self.second_layer_model.fit(predictions, new_y)

    def predict(self, X):
        predictions = None
        for m in self.models:
            p = m.multi_predict(X)
            if predictions is None:
                predictions = p
            else:
                predictions += p

        predictions /= len(self.models)

        return self.second_layer_model.predict(predictions)

    def get_params(self, deep = False):
        return { 'alpha': 1 }