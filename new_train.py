#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import ensemble, cross_validation
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from config import data_prefix, path_prefix

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
    # model = Pipeline([('poly', PolynomialFeatures(degree=2)),
    #                   ('ridge', linear_model.Ridge(alpha=1100, copy_X=True))])
    model = linear_model.Lasso(alpha=0.0001, copy_X=True)
    model = linear_model.Ridge(alpha=1, copy_X= True)
    model = ensemble.RandomForestRegressor(n_estimators=500)
                      # ('lasso', linear_model.Lasso(alpha=0.0005, copy_X=True))])
    # # # model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
    # # #                   ('svr', svm.SVR())])
    cv_res = cross_validation.cross_val_score(model, X, y, cv = 5, scoring = 'neg_mean_squared_error')
    print("\nscore: {:.5f} ({:.5f})\n".format(np.mean(np.sqrt(-cv_res)), np.std(np.sqrt(-cv_res))))

    # train model
    model.fit(X, y)

    # output to files
    joblib.dump(model, data_prefix + 'model.m')
    correlation.to_csv(data_prefix + 'corr.csv')