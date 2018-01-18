from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
from preprocess_funs import *
from config import path_prefix, data_prefix
from scipy.special import boxcox1p

bsmtFinType_df = Series([0, 1, 2, 3, 4, 5, 6], index=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], dtype = int)

quality_df = Series([0, 1, 2, 3, 4, 5], index=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], dtype = int)

garage_finish_df = Series([0, 1, 2, 3], index=['NA', 'Unf', 'RFn', 'Fin'], dtype = int)

fence_df = Series([0, 1, 2, 3, 4], index=['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], dtype = int)

exposure_df = Series([0, 1, 2, 3, 4], index=['NA', 'No', 'Mn', 'Av', 'Gd'], dtype = int)

centralAir_df = Series([0, 1], index=['N', 'Y'], dtype = int)

paved_drive_df = Series([0, 1, 2], index=['N', 'P', 'Y'], dtype = int)

train = pd.read_csv(data_prefix + 'train.csv')
test = pd.read_csv(data_prefix + 'test.csv')

train.drop(['SalePrice'], axis=1, inplace=True)
train_index = train.index

data = pd.concat([train, test])
data.index = range(len(data))

#### fill null or NA
# PoolQC fill
index_pool = data.loc[(data.PoolArea != 0) & data.PoolQC.isnull(), 'PoolQC'].index
data.loc[index_pool, 'PoolQC'] = Series(['Ex', 'Ex', 'Fa'], index=index_pool)

# GarageBlt fill
index_garageBlt = data.loc[data.GarageYrBlt.isnull()].index
data.loc[index_garageBlt, 'GarageYrBlt'] = data.loc[index_garageBlt, 'YearBuilt']

# Garage* fill
data.GarageCond.fillna('NA', inplace=True)
data.GarageQual.fillna('NA', inplace=True)
data.GarageType.fillna('NA', inplace=True)
data.GarageFinish.fillna('NA', inplace=True)
data.GarageArea.fillna(0, inplace=True)
data.GarageCars.fillna(0, inplace=True)

# Kitchen* fill
kitchen_null = data.KitchenAbvGr[data.KitchenQual.isnull()].values[0]
most_frequent_kitchen = \
data.KitchenQual[data.KitchenAbvGr == kitchen_null].value_counts().sort_values(ascending=False).index[0]
data.KitchenQual[data.KitchenQual.isnull()] = most_frequent_kitchen

# Electrical fill
most_frequent_eletrical = data.Electrical.value_counts().sort_values(ascending=False).index[0]
data.Electrical[data.Electrical.isnull()] = most_frequent_eletrical

# BsmtSF fill
data.loc[data.TotalBsmtSF.isna(), ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']] = 0

# BsmtType1 fill
data.loc[(data.BsmtFinType1.isnull()) & (data.BsmtFinSF1 == 0), 'BsmtFinType1'] = 'NA'

# BsmtType2 fill
data.loc[(data.BsmtFinType2.isnull()) & (data.BsmtFinSF2 == 0), 'BsmtFinType2'] = 'NA'
data.loc[(data.BsmtFinType2.isnull()) & (data.BsmtFinSF2 == 479), 'BsmtFinType2'] = 'ALQ'

# BsmtQual fill
data.BsmtQual.fillna('NA', inplace=True)

# BsmtExposure fill
data.loc[data.TotalBsmtSF == 0, 'BsmtExposure'] = 'NA'
data.loc[data.BsmtExposure.isnull(), 'BsmtExposure'] = ['No', 'Gd', 'No']

# TotalBsmtSF fill
data.loc[data.TotalBsmtSF == 0, 'BsmtCond'] = 'NA'
data.loc[data.BsmtCond.isnull(), 'BsmtCond'] = ['Gd', 'TA', 'Po']

# Exterior fill
data.Exterior1st.fillna('NA', inplace=True)
data.Exterior2nd.fillna('NA', inplace=True)

# SaleType fill
data.loc[data.SaleType.isnull(), 'SaleType'] = 'WD'

# Functional fill
data.loc[data.Functional.isnull(), 'Functional'] = 'Typ'

# Utilities fill
data.loc[data.Utilities.isnull(), 'Utilities'] = 'AllPub'

# MSZoning fill
data.loc[data.MSZoning.isnull(), 'MSZoning'] = ['RM', 'RL', 'RM', 'RL']

# BsmtFullBath fill
data.BsmtFullBath.fillna(0, inplace=True)

# BsmtHalfBath fill
data.BsmtHalfBath.fillna(0, inplace=True)

# Mas fill
data.loc[(data.MasVnrArea.isna()) & (data.MasVnrType.isnull()), 'MasVnrArea'] = 0
data.loc[(data.MasVnrArea.isna()) & (data.MasVnrType.isnull()), 'MasVnrType'] = 'NA'
data.MasVnrType.fillna('BrkCmn', inplace=True)

# Lot fill
lot_median = data[['Neighborhood', 'LotFrontage']].groupby('Neighborhood').median()
neighborhood = data.loc[data.LotFrontage.isna(), 'Neighborhood']
data.loc[data.LotFrontage.isna(), 'LotFrontage'] = lot_median.loc[neighborhood].values

# FireplaceQu fill
data.FireplaceQu.fillna('NA', inplace=True)

# Fence fill
data.Fence.fillna('NA', inplace=True)

# Alley
data.Alley.fillna('NA', inplace=True)

# PoolQC fill
data.PoolQC.fillna('NA', inplace=True)

# MiscFeature fill
data.MiscFeature.fillna('NA', inplace=True)

#### numrical
data.ExterQual = quality_df[data.ExterQual.values].values
data.BsmtQual = quality_df[data.BsmtQual.values].values
data.HeatingQC = quality_df[data.HeatingQC.values].values
data.KitchenQual = quality_df[data.KitchenQual.values].values
data.FireplaceQu = quality_df[data.FireplaceQu.values].values
data.GarageQual = quality_df[data.GarageQual.values].values
data.PoolQC = quality_df[data.PoolQC.values].values
data.ExterCond = quality_df[data.ExterCond.values].values
data.BsmtCond = quality_df[data.BsmtCond.values].values
data.GarageCond = quality_df[data.GarageCond.values].values
data.BsmtFinType1 = bsmtFinType_df[data.BsmtFinType1.values].values
data.BsmtFinType2 = bsmtFinType_df[data.BsmtFinType2.values].values
data.GarageFinish = garage_finish_df[data.GarageFinish.values].values
data.Fence = fence_df[data.Fence.values].values
data.BsmtExposure = exposure_df[data.BsmtExposure.values].values
data.CentralAir = centralAir_df[data.CentralAir.values].values
data.PavedDrive = paved_drive_df[data.PavedDrive.values].values

# log process
# data.LotArea = np.log1p(data.LotArea)
# data.OpenPorchSF = np.log1p(data.OpenPorchSF)
# data.WoodDeckSF = np.log1p(data.WoodDeckSF)
# data.LotFrontage = np.log1p(data.LotFrontage)
# data['1stFlrSF'] = np.log1p(data['1stFlrSF'])
# data.MasVnrArea = np.log1p(data.MasVnrArea)
# data.BsmtFinSF2 = np.log1p(data.BsmtFinSF2)
# data.BsmtUnfSF = np.log1p(data.BsmtUnfSF)
# data.TotalBsmtSF = np.log1p(data.TotalBsmtSF)
# data.GrLivArea = np.log1p(data.GrLivArea)
# data.GarageArea = np.log1p(data.GarageArea)
# data.ScreenPorch = np.log1p(data.ScreenPorch)

index = [u'LotFrontage', u'LotArea', u'YearBuilt', u'YearRemodAdd', u'MasVnrArea',
       u'BsmtFinSF1', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'1stFlrSF',
       u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea',
       u'KitchenAbvGr', u'TotRmsAbvGrd', u'Fireplaces',
       u'FireplaceQu', u'GarageYrBlt', u'GarageCars', u'GarageArea',
       u'WoodDeckSF', u'OpenPorchSF', u'EnclosedPorch', u'3SsnPorch',
       u'ScreenPorch', u'PoolArea', u'MiscVal', u'MoSold', u'YrSold']
skewness = data[index].skew().sort_values()
features = skewness[np.abs(skewness) > 0.75].index
lam = 0.15
for feat in features:
    data[feat] = boxcox1p(data[feat], lam)

# extra features
data['Neighborhood_rich'] = 0
data['Neighborhood_rich'] = data.Neighborhood.apply(lambda neighbor: 1 if neighbor in [u'StoneBr', u'NoRidge', u'NridgHt'] else 0)
data['hasPool'] = data['PoolArea'].apply(lambda x: 0 if x == 0 else 1)
data['has3SsnPorch'] = data['3SsnPorch'].apply(lambda x: 0 if x == 0 else 1)
data['hasEnclosedPorch'] = data['EnclosedPorch'].apply(lambda x: 0 if x == 0 else 1)
data['hasGarage'] = data['GarageFinish'].apply(lambda x: 0 if x == 0 else 1)
data['hasMasVnrArea'] = data['MasVnrArea'].apply(lambda x: 0 if x == 0 else 1)
data['hasOpenPorchSF'] = data['OpenPorchSF'].apply(lambda x: 0 if x == 0 else 1)
data['hasWoodDeckSF'] = data['WoodDeckSF'].apply(lambda x: 0 if x == 0 else 1)
data['hasBsmtUnfSF'] = data['BsmtUnfSF'].apply(lambda x: 0 if x == 0 else 1)
data['is2ndFlrSFZero'] = data['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
data['hasBsmtFinSF1'] = data['BsmtFinSF1'].apply(lambda x: 0 if x == 0 else 1)
data['hasBsmtFinSF2'] = data['BsmtFinSF2'].apply(lambda x: 0 if x == 0 else 1)
data['hasBsmtUnfSF'] = data['BsmtUnfSF'].apply(lambda x: 0 if x == 0 else 1)
data['hasTotalBsmtSF'] = data['TotalBsmtSF'].apply(lambda x: 0 if x == 0 else 1)
data['hasGarageArea'] = data['GarageArea'].apply(lambda x: 0 if x == 0 else 1)
data['hasScreenPorch'] = data['ScreenPorch'].apply(lambda x: 0 if x == 0 else 1)
data['isOverallQualLow'] = data['OverallQual'].apply(lambda x: 1 if x <= 2 else 0)
data['YrSoldBuket'] = pd.cut(data.YrSold, 10, labels=range(10))
data['YearBuiltBuket'] = pd.cut(data.YearBuilt, 10, labels=range(10))
data['GarageYrBltBuket'] = pd.cut(data.GarageYrBlt, 10, labels=range(10))

# standardize
standardizing_features = [u'LotFrontage', u'LotArea', u'YearBuilt', u'YearRemodAdd', u'MasVnrArea',
       u'BsmtFinSF1', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'1stFlrSF',
       u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea',
       u'KitchenAbvGr', u'TotRmsAbvGrd', u'Fireplaces',
       u'FireplaceQu', u'GarageYrBlt', u'GarageCars', u'GarageArea',
       u'WoodDeckSF', u'OpenPorchSF', u'EnclosedPorch', u'3SsnPorch',
       u'ScreenPorch', u'PoolArea', u'MiscVal', u'MoSold',
       u'YrSold']
data.loc[:, standardizing_features], scaler = normalization(data.loc[:, standardizing_features])

# data.drop(['YrSold', 'YearBuilt', 'GarageYrBlt'], axis = 1, inplace = True)
## dummy
categoric_features = [u'MSZoning', u'Street', u'Alley', u'LotShape', u'LandContour',
                      u'Utilities', u'LotConfig', u'LandSlope', u'Neighborhood',
                      u'Condition1', u'Condition2', u'BldgType', u'HouseStyle', u'RoofStyle',
                      u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
                      u'Foundation', u'Heating', u'Electrical', u'Functional',
                      u'GarageType', u'MiscFeature', u'SaleType',
                      u'SaleCondition', u'OverallCond', u'YrSoldBuket', u'YearBuiltBuket', u'GarageYrBltBuket', u'MSSubClass']

for feature in categoric_features:
    dummy_feature = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, dummy_feature], axis=1)
    data.drop([feature], axis=1, inplace=True)

# drop Id
data.drop(['Id'], axis = 1, inplace = True)

# split train / test set
train = pd.read_csv(data_prefix + 'train.csv')
new_train = data.loc[train_index]
new_train['SalePrice'] = train['SalePrice']

new_train = new_train[train.GrLivArea <= 4000]
new_test = data.drop(train_index)

# save data
new_train.to_csv(data_prefix + 'preprocessed_train.csv')
new_test.to_csv(data_prefix + 'preprocessed_test.csv')