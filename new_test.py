#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.externals import joblib

if __name__ == '__main__':
    path_prefix = '/Users/xueweiyao/Documents/house price/'
    correlation = pd.read_csv('/Users/xueweiyao/Documents/house price/corr.csv')
    model = joblib.load(path_prefix + 'model.m')
    data = pd.read_csv('/Users/xueweiyao/Documents/house price/preprocessed_test.csv')

    # feature selection
    correlation.index = correlation.residual
    corr_index = correlation.residual.drop(['SalePrice'])
    data = data[corr_index]

    # predict
    test_X = data.as_matrix()
    prediction = model.predict(test_X)
    prediction = np.reshape(prediction, np.shape(prediction)[0])
    prediction = np.expm1(prediction)

    res = DataFrame({'SalePrice': prediction}, index=range(1461, 1461 + len(prediction)))
    res.index.name = 'Id'

    res.to_csv(path_prefix + 'prediction.csv')
