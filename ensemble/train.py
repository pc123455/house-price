#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import ensemble, model_selection
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from config import data_prefix, path_prefix
from ensemble.models import EnsenbleModel

def rmse(test, predict):
    return np.sqrt(np.sum(np.square(predict - test)) / len(test))

def score(model, X, y):
    prediction = model.predict(X)
    return np.sum(np.square(np.expm1(prediction) - np.expm1(y)))

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
    data = data[indecies]

    # cross validation
    X = data.drop(['SalePrice'], axis=1).fillna(0).as_matrix()
    y = np.log1p(data['SalePrice'].as_matrix())

    models = EnsenbleModel()
    cv_res = model_selection.cross_val_score(models, X, y, fit_params={}, cv = 5, scoring = 'neg_mean_squared_error')
    print("\nscore: {:.5f} ({:.5f})\n".format(np.mean(np.sqrt(-cv_res)), np.std(np.sqrt(-cv_res))))


    # train model
    models.fit(X, y)

    # output to files
    joblib.dump(models, data_prefix + 'models.m')
    correlation.to_csv(data_prefix + 'corr.csv')