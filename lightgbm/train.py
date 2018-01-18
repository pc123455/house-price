#!/usr/bin/python
# -*- coding: utf-8 -*-

import lightgbm as lgb


model = lgb.LGBMRegressor(boosting_type = 'gbdt',
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


