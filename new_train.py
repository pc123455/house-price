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
    return np.sqrt(np.sum(np.square(prediction - test_y)) / len(test))

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

    # cross validation split
    split_train, split_test = cross_validation.train_test_split(data, test_size=0.3, random_state=0)

    # train
    train = split_train
    test = split_test

    train_X = train.drop(['SalePrice'], axis=1).fillna(0).as_matrix()
    train_y = train['SalePrice'].as_matrix()
    train_y = np.log1p(train_y)

    test_X = test.drop(['SalePrice'], axis=1).fillna(0).as_matrix()
    test_y = test['SalePrice'].as_matrix()

    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('ridge', linear_model.Ridge(alpha=2900, copy_X=True))])
                      # ('ridge', linear_model.Lasso(alpha=0.0112, copy_X=True))])
    # # model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
    # #                   ('svr', svm.SVR())])
    # #model = linear_model.LassoLars(alpha=0.01, copy_X=True)
    model.fit(train_X, train_y)

    prediction = model.predict(test_X)
    prediction = np.reshape(prediction, np.shape(prediction)[0])
    prediction = np.expm1(prediction)
    # test_y = np.reshape(test_y, np.shape(test_y)[0])

    print rmse(test_y, prediction)
    print np.sum(np.square(prediction - test_y))
    joblib.dump(model, data_prefix + 'model.m')
    correlation.to_csv(data_prefix + 'corr.csv')