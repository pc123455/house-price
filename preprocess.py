from preprocess_funs import *
from sklearn.externals import joblib
import numpy as np

if __name__ == "__main__":
    prefix_path = '/Users/xueweiyao/Documents/house price/'
    data = pd.read_csv(prefix_path + 'train.csv')

    data = remove_bit(data)
    data = fill_null(data)
    data, Electrical_default_value = set_null_value_mode(data, 'Electrical')
    data, MasVnrType_default_value = set_null_value_mode(data, 'MasVnrType')

    data = discrete_dummy(data)
    data, rfr_lot = set_LotFrontage(data)
    data, rfr_mas = set_MasVnrArea(data)
    data, rfr_gar = set_GarageYrBlt(data)

    data, rfc_BsmtQual = set_null_value_rfc(data, 'BsmtQual_.*')
    data, rfc_BsmtCond = set_null_value_rfc(data, 'BsmtCond_.*')
    data, rfc_BsmtExposure = set_null_value_rfc(data, 'BsmtExposure_.*')
    data, rfc_BsmtFinType1 = set_null_value_rfc(data, 'BsmtFinType1_.*')
    data, rfc_BsmtFinType2 = set_null_value_rfc(data, 'BsmtFinType2_.*')
    data, rfc_FireplaceQu = set_null_value_rfc(data, 'FireplaceQu_.*')
    data, rfc_GarageType = set_null_value_rfc(data, 'GarageType_.*')
    data, rfc_GarageFinish = set_null_value_rfc(data, 'GarageFinish_.*')
    data, rfc_GarageQual = set_null_value_rfc(data, 'GarageQual_.*')
    data, rfc_GarageCond = set_null_value_rfc(data, 'GarageCond_.*')

    # log transform

    # data.loc['LotArea'] = np.log1p(data['LotArea'])
    # data.loc['TotalBsmtSF'] = np.log1p(data['TotalBsmtSF'])
    # data.loc['GrLivArea'] = np.log1p(data['GrLivArea'])
    # data.loc['GarageArea'] = np.log1p(data['GarageArea'])
    # data.loc['WoodDeckSF'] = np.log1p(data['WoodDeckSF'])
    # data.loc['OpenPorchSF'] = np.log1p(data['OpenPorchSF'])
    # data['OtherPorch'] = np.log1p(data[['EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis = 1))
    # data.loc['PoolArea'] = np.log1p(data['PoolArea'])
    # data.loc['MiscVal'] = np.log1p(data['MiscVal'])
    # data.loc['LotFrontage'] = np.log1p(data['LotFrontage'])
    # data.loc['MasVnrArea'] = np.log1p(data['MasVnrArea'])
    # data.loc['GarageYrBlt'] = np.log1p(data['GarageYrBlt'])


    # scaling
    regex = 'LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|BsmtFinSF1|\
             BsmtFinSF2|BsmtUnfSF|TotalBsmtSF|1stFlrSF|2ndFlrSF|LowQualFinSF|\
             GrLivArea|BsmtFullBath|BsmtHalfBath|FullBath|HalfBath|BedroomAbvGr|\
             KitchenAbvGr|TotRmsAbvGrd|Fireplaces|GarageCars|GarageArea|WoodDeckSF|\
             OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|PoolArea|\
             MiscVal|MoSold|YrSold|LotFrontage|MasVnrArea|GarageYrBlt'
    regex = regex.replace(' ', '')
    data_float = data.filter(regex = regex)
    data_float, scaler = normalization(data_float)
    data[data_float.columns] = data_float

    # feature selection
    corr_thres = 0.01
    corrmat = data.corr()
    sale_corr = corrmat[['SalePrice']]
    indecies = sale_corr[sale_corr['SalePrice'] > corr_thres]

    # out to file
    joblib.dump(rfr_lot, prefix_path + 'rfr_lot.m')
    joblib.dump(rfr_mas, prefix_path + 'rfr_mas.m')
    joblib.dump(rfr_gar, prefix_path + 'rfr_gar.m')
    joblib.dump(scaler, prefix_path + 'standard_scaler.m')
    data.to_csv('/Users/xueweiyao/Documents/house price/prepared.csv')
    indecies.to_csv('/Users/xueweiyao/Documents/house price/correlation.csv')