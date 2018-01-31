#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from config import data_prefix
from ensemble.models import EnsenbleModel

if __name__ == '__main__':
    data = pd.read_csv(data_prefix + 'preprocessed_train.csv')

    # feature selection
    # filter
    corr_thres = 0.1
    corrmat = data.corr()
    sale_corr = corrmat[['SalePrice']]
    correlation = sale_corr[np.abs(sale_corr['SalePrice']) > corr_thres]
    correlation.index.name = 'residual'
    indecies = correlation.index
    # SalePrice.sort_values(ascending=False)
    data = data[indecies]

    # cross validation
    X = data.drop(['SalePrice'], axis=1).as_matrix()
    y = np.log1p(data['SalePrice'].as_matrix())

    model = EnsenbleModel()

    # cross validation
    kf = KFold(n_splits = 5,shuffle = False)

    # models
    ridges = list()
    lassos = list()
    rfrs = list()

    # 1st layer predictions
    predict_ridge = np.zeros((len(y), 1), dtype = float)
    predict_lasso = np.zeros((len(y), 1), dtype = float)
    predict_rfr = np.zeros((len(y), 1), dtype = float)

    # 1st layer
    for train_index, test_index in kf.split(X):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]

        # ridge regression with degree 2
        ridge_model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
                                ('ridge', linear_model.Ridge(alpha=300))])
        ridge_model.fit(train_X, train_y)
        predict_ridge[test_index, 0] = ridge_model.predict(test_X)
        ridges.append(ridge_model)

        # lasso regression with degree 2
        lasso_model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
                                ('lasso', linear_model.Lasso(alpha=100))])
        lasso_model.fit(train_X, train_y)
        predict_lasso[test_index, 0] = lasso_model.predict(test_X)
        lassos.append(lasso_model)

        # random forest
        rfr_model = RandomForestRegressor(random_state = 1, n_estimators = 100, n_jobs = -1)
        rfr_model.fit(train_X, train_y)
        predict_rfr[test_index, 0] = rfr_model.predict(test_X)
        rfrs.append(rfr_model)

    # 2nd layer
    second_layer_model = linear_model.Ridge(alpha = 1)
    second_layer_X = np.concatenate((predict_ridge, predict_lasso, predict_rfr), axis=1)
    second_layer_model.fit(second_layer_X, y)

    # dump to file
    joblib.dump(ridges, path_prefix + 'ridges.m')
    joblib.dump(lassos, path_prefix + 'lassos.m')
    joblib.dump(rfrs, path_prefix + 'rfrs.m')
    joblib.dump(second_layer_model, path_prefix + 'final_model.m')