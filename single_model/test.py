#!/usr/bin/python
# -*- coding: utf-8 -*-
from pandas import DataFrame, Series
import numpy as np
from preprocess_funs import *
from sklearn.externals import joblib
import pandas as pd

# load data
path_prefix = '/Users/xueweiyao/Documents/house price/'
correlation = pd.read_csv('/Users/xueweiyao/Documents/house price/corr.csv')
model = joblib.load(path_prefix +'model.m')
rfr_LotFrontage = joblib.load(path_prefix + 'rfr_lot.m')
rfr_MasVnrArea = joblib.load(path_prefix + 'rfr_mas.m')
rfr_GarageYrBlt = joblib.load(path_prefix + 'rfr_gar.m')
scaler = joblib.load(path_prefix + 'standard_scaler.m')

data_test = pd.read_csv(path_prefix + 'test.csv')
data_test['SalePrice'] = np.nan
data_train = pd.read_csv(path_prefix + 'train.csv')
data = pd.concat([data_train, data_test])

data, Electrical_default_value = set_null_value_mode(data, 'Electrical', Electrical_default_value)
data, MasVnrType_default_value = set_null_value_mode(data, 'MasVnrType', MasVnrType_default_value)

data = discrete_dummy(data)
data, rfr_lot = set_LotFrontage(data, rfr_lot)
data, rfr_mas = set_MasVnrArea(data, rfr_mas)
data, rfr_gar = set_GarageYrBlt(data, rfr_gar)

data, rfc_BsmtQual = set_null_value_rfc(data, 'BsmtQual_.*', rfc_BsmtQual)
data, rfc_BsmtCond = set_null_value_rfc(data, 'BsmtCond_.*', rfc_BsmtCond)
data, rfc_BsmtExposure = set_null_value_rfc(data, 'BsmtExposure_.*', rfc_BsmtExposure)
data, rfc_BsmtFinType1 = set_null_value_rfc(data, 'BsmtFinType1_.*', rfc_BsmtFinType1)
data, rfc_BsmtFinType2 = set_null_value_rfc(data, 'BsmtFinType2_.*', rfc_BsmtFinType2)
data, rfc_FireplaceQu = set_null_value_rfc(data, 'FireplaceQu_.*', rfc_FireplaceQu)
data, rfc_GarageType = set_null_value_rfc(data, 'GarageType_.*', rfc_GarageType)
data, rfc_GarageFinish = set_null_value_rfc(data, 'GarageFinish_.*', rfc_GarageFinish)
data, rfc_GarageQual = set_null_value_rfc(data, 'GarageQual_.*', rfc_GarageQual)
data, rfc_GarageCond = set_null_value_rfc(data, 'GarageCond_.*', rfc_GarageCond)

regex_non_num = 'MSZoning_.*|Street_.*|LotShape_.*|LandContour_.*|\
             LotConfig_.*|LandSlope_.*|Foundation_.*|\
             Neighborhood_.*|Condition1_.*|Condition2_.*|BsmtQual_.*|BsmtCond_.*|\
             BldgType_.*|HouseStyle_.*|RoofStyle_.*|RoofMatl_.*|\
             Exterior1st_.*|Exterior2nd_.*|MasVnrType_.*|ExterQual_.*|\
             ExterCond_.*|BsmtExposure_.*|BsmtFinType1_.*|BsmtFinType2_.*|\
             Heating_.*|HeatingQC_.*|CentralAir_.*|Electrical_.*|\
             KitchenQual_.*|Functional_.*|FireplaceQu_.*|GarageType_.*|\
             GarageFinish_.*|GarageQual_.*|GarageCond_.*|PavedDrive_.*|\
             PoolQC_.*|Fence_.*|MiscFeature_.*|SaleType_.*|SaleCondition_.*|MSSubClass'
regex_non_num = regex_non_num.replace(' ', '')

regex_num = 'LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|BsmtFinSF1|\
             BsmtFinSF2|BsmtUnfSF|TotalBsmtSF|1stFlrSF|2ndFlrSF|LowQualFinSF|\
             GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|HalfBath|BedroomAbvGr|\
             KitchenAbvGr|TotRmsAbvGrd|Fireplaces|GarageCars|GarageArea|WoodDeckSF|\
             OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|PoolArea|\
             MiscVal|MoSold|YrSold|LotFrontage|MasVnrArea|GarageYrBlt'
regex_num = regex_num.replace(' ', '')

test = data[data.SalePrice.isna()]
df = test.filter(regex = regex_non_num)
df_float = test.filter(regex = regex_num).fillna(0)

data = pd.concat([df, df_float], axis = 1)
data = data.reindex_axis(sorted(data.columns), axis = 1)

# feature selection
correlation.index = correlation.residual
corr_index = correlation.residual.drop(['SalePrice'])
data = data[corr_index]

# scaling
regex = 'LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|BsmtFinSF1|\
         BsmtFinSF2|BsmtUnfSF|TotalBsmtSF|1stFlrSF|2ndFlrSF|LowQualFinSF|\
         GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|HalfBath|BedroomAbvGr|\
         KitchenAbvGr|TotRmsAbvGrd|Fireplaces|GarageCars|GarageArea|WoodDeckSF|\
         OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|PoolArea|\
         MiscVal|MoSold|YrSold|LotFrontage|MasVnrArea|GarageYrBlt'.replace(' ', '')
data_float = data.filter(regex = regex)
data_float, scaler = normalization(data_float, scaler)
data[data_float.columns] = data_float

# predict
test_X = data.as_matrix()
prediction = model.predict(test_X)
prediction = np.reshape(prediction, np.shape(prediction)[0])

res = DataFrame({'SalePrice' : prediction}, index = range(1461, 1461 + len(prediction)))
res.index.name = 'Id'

res.to_csv(path_prefix + 'prediction.csv')