#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
from config import data_prefix
from ensemble.models import EnsembleModel
import numpy as np
from sklearn.externals import joblib

if __name__ == '__main__':
    correlation = pd.read_csv(data_prefix + 'corr.csv')
    models = joblib.load(data_prefix + 'models.m')
    data = pd.read_csv(data_prefix + 'preprocessed_test.csv')

    # feature selection
    correlation.index = correlation.residual
    corr_index = correlation.residual.drop(['SalePrice'])
    data = data[corr_index]

    # predict
    test_X = data.as_matrix()
    prediction = models.predict(test_X)
    prediction = np.reshape(prediction, np.shape(prediction)[0])
    prediction = np.expm1(prediction)

    res = DataFrame({'SalePrice': prediction}, index=range(1461, 1461 + len(prediction)))
    res.index.name = 'Id'

    res.to_csv(data_prefix + 'prediction.csv')