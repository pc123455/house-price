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
from sklearn.feature_selection import RFE
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/xueweiyao/Documents/house price/prepared.csv')

regex_non_num = 'SalePrice|MSZoning_.*|Street_.*|LotShape_.*|\
             LotConfig_.*|LandSlope_.*|LandContour_.*|Foundation_.*|\
             Neighborhood_.*|Condition1_.*|Condition2_.*|BsmtQual_.*|BsmtCond_.*|\
             BldgType_.*|HouseStyle_.*|RoofStyle_.*|RoofMatl_.*|\
             Exterior1st_.*|Exterior2nd_.*|MasVnrType_.*|ExterQual_.*|\
             ExterCond_.*|BsmtExposure_.*|BsmtFinType1_.*|BsmtFinType2_.*|\
             Heating_.*|HeatingQC_.*|CentralAir_.*|Electrical_.*|\
             KitchenQual_.*|Functional_.*|FireplaceQu_.*|GarageType_.*|\
             GarageFinish_.*|GarageQual_.*|GarageCond_.*|PavedDrive_.*|\
             PoolQC_.*|Fence_.*|MiscFeature_.*|SaleType_.*|SaleCondition_.*|PoolQC_.*|MSSubClass'
regex_non_num = regex_non_num.replace(' ', '')

regex_num = 'LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|\
             TotalBsmtSF|1stFlrSF|2ndFlrSF|LowQualFinSF|\
             GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|HalfBath|BedroomAbvGr|\
             KitchenAbvGr|TotRmsAbvGrd|Fireplaces|GarageCars|GarageArea|WoodDeckSF|\
             OpenPorchSF|OtherPorch|PoolArea|\
             MiscVal|MoSold|YrSold|LotFrontage|MasVnrArea|GarageYrBlt'

regex_num = regex_num.replace(' ', '')
df = data.filter(regex = regex_non_num)
df_float = data.filter(regex = regex_num)

df_float = df_float.fillna(0)
data = pd.concat([df, df_float], axis = 1)
data = data.reindex_axis(sorted(data.columns), axis = 1)

# feature selection
# filter
corr_thres = 0.15
corrmat = data.corr()
sale_corr = corrmat[['SalePrice']]
correlation = sale_corr[np.abs(sale_corr['SalePrice']) > corr_thres]
correlation.index.name = 'residual'
indecies = correlation.index
data = data[indecies]

# cross validation split
split_train, split_test = cross_validation.train_test_split(data, test_size = 0.3, random_state=0)

# train
train = split_train
test = split_test

train_X = train.drop(['SalePrice'], axis = 1).fillna(0).as_matrix()
train_y = train[['SalePrice']].as_matrix()
train_y = np.log1p(train_y)

test_X = test.drop(['SalePrice'], axis = 1).fillna(0).as_matrix()
test_y = test[['SalePrice']].as_matrix()

#wrapper
# rfe = RFE(estimator = Pipeline([('poly', PolynomialFeatures(degree = 2)),
#                   ('ridge', linear_model.Lasso(alpha = 2000, copy_X = True, tol=1))]), n_features_to_select = 90)
# rfe.fit(train_X, train_y)
# prediction = rfe.predict(test_X)

# model = ensemble.RandomForestRegressor(random_state = 1, n_estimators = 10, n_jobs = -1)
model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
                  ('ridge', linear_model.Ridge(alpha = 400, copy_X = True))])
# # model = Pipeline([('poly', PolynomialFeatures(degree = 2)),
# #                   ('svr', svm.SVR())])
# #model = linear_model.LassoLars(alpha=0.01, copy_X=True)
model.fit(train_X, train_y)

prediction = model.predict(test_X)
prediction = np.reshape(prediction, np.shape(prediction)[0])
prediction = np.exp(prediction) - 1
test_y = np.reshape(test_y, np.shape(test_y)[0])

print np.sum(np.square(prediction - test_y))

price = DataFrame({'pridiction' : prediction, 'y' : test_y, 'diff' : prediction - test_y})
# price.plot()
# plt.show()

joblib.dump(model, '/Users/xueweiyao/Documents/house price/model.m')
correlation.to_csv('/Users/xueweiyao/Documents/house price/corr.csv')