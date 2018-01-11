#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('/Users/xueweiyao/Documents/house price/preprocessed_train.csv')



# feature selection
corr_thres = 0.15
corrmat = data.corr()
sale_corr = corrmat[['SalePrice']]
correlation = sale_corr[np.abs(sale_corr['SalePrice']) > corr_thres]
correlation.index.name = 'residual'
data = data[correlation.index]

# cross validation split
split_train, split_test = cross_validation.train_test_split(data, test_size = 0.3, random_state = 0)

# train
train = split_train
test = split_test

train_X = train.drop(['SalePrice'], axis = 1).as_matrix()
train_y = train['SalePrice'].as_matrix()

test_X = test.drop(['SalePrice'], axis = 1).as_matrix()
test_y = test['SalePrice'].as_matrix()

poly = PolynomialFeatures(2)
train_X = poly.fit_transform(train_X)
test_X = poly.fit_transform(test_X)

# create dataset for lightgbm

lgb_train = lgb.Dataset(train_X, np.log1p(train_y)*100)
lgb_eval = lgb.Dataset(test_X, np.log1p(test_y)*100, reference = lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'max_bin': 128,
    'num_trees': 5000,
    'num_leaves': 1024,
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'verbose': 0,
    'num_threads': 2,
    'lambda_l1': 0.5,
    'min_data_in_leaf': 10,
    'bagging_seed':3,
    'early_stopping_round': 100,
    'min_data_per_group': 20
}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
print('Save model...')
# save model to file
gbm.save_model('model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(test_X, num_iteration = gbm.best_iteration)
y_pred = np.expm1(y_pred / 100.0)
# eval
print(y_pred)
print('Dump model to JSON...')
# dump model to json (and save to file)

# model_json = gbm.dump_model()
# with open('model.json', 'w+') as f:
#     json.dump(model_json, f, indent=4)

print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

print np.sum(np.square(y_pred - test_y))
price = DataFrame({'pridiction' : y_pred, 'y' : test_y, 'diff' : y_pred - test_y})
# price.plot()
plt.scatter(y_pred, test_y)
plt.show()
