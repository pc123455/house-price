#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import  preprocessing

def remove_bit(data):
    """remove columns which include notnull value"""
    data.drop(['Alley', 'Utilities', 'MiscFeature', 'Fence'], axis=1, inplace=True)
    return data

def fill_null(data):
    """fill cell which is null"""
    data.loc[data['PoolQC'].isnull(), 'PoolQC'] = 'null'
    return data

def set_null_value_mode(data, col_name, value = None):
    if value == None:
        value = data[col_name].value_counts()[0]

    data.loc[data[col_name].isnull(), col_name] = value
    return data, value

def set_null_value_rfc(data, col_regex, rfc = None):
    """setting discrete value using other columns"""
    regex = 'MSZoning_.*|Street_.*|LotShape_.*|LandContour_.*|LotConfig_.*|\
       LandSlope_.*|Neighborhood_.*|Condition1_.*|Condition2_.*|\
       BldgType_.*|HouseStyle_.*|RoofStyle_.*|RoofMatl_.*|Exterior1st_.*|\
       Exterior2nd_.*|ExterQual_.*|ExterCond_.*|Foundation_.*|Heating_.*|\
       HeatingQC_.*|CentralAir_.*|Electrical_.*|KitchenQual_.*|\
       Functional_.*|PavedDrive_.*|SaleType_.*|SaleCondition_.*|\
       MSSubClass|LotArea|OverallQual|OverallCond|\
       YearBuilt|YearRemodAdd|BsmtFinSF1|BsmtFinSF2|\
       BsmtUnfSF|TotalBsmtSF|1stFlrSF|2ndFlrSF|LowQualFinSF|\
       GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|\
       HalfBath|BedroomAbvGr|KitchenAbvGr|TotRmsAbvGrd|\
       Fireplaces|GarageCars|GarageArea|WoodDeckSF|\
       OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|\
       PoolArea|MiscVal|MoSold|YrSold'
    regex = regex.replace(' ', '')

    null_cols = data.filter(regex = col_regex)
    df = data.filter(regex = regex)
    df = df.fillna(0)

    train_Y = null_cols[(null_cols != 0).any(axis=1)].as_matrix()
    train_X = df[(null_cols != 0).any(axis=1)].as_matrix()
    test_X = df[(null_cols == 0).all(axis=1)].as_matrix()

    if rfc is None:
        rfc = RandomForestClassifier(random_state=0, n_estimators=200, n_jobs= -1)
        rfc.fit(train_X, train_Y)

    test_Y = rfc.predict(test_X)

    data.loc[(null_cols == 0).all(axis=1), null_cols.columns] = test_Y

    return data, rfc

def normalization(data, scaler = None):
    data = data.fillna(0)
    X = data.as_matrix()
    if scaler is None:
        scaler = preprocessing.RobustScaler()
        scaler.fit(X)

    data[data.columns] = scaler.fit_transform(X)

    return data, scaler

def move_price_to_end(data):
    price = data['SalePrice']
    data.drop(['SalePrice'], axis=1, inplace = True)
    data['SalePrice'] = price

    return data