import rqdatac
from rqdatac import *
rqdatac.init('ricequant', '8ricequant8')
import pandas as pd

class OptimizationError(Exception):
    def __init__(self, warning_message):
        print(warning_message)



def input_validation(order_book_ids, rebalancing_date, asset_type, method, window, bnds,
                     cons, cov_shrinkage, benchmark, expected_return, industry_matching, risk_aversion_coefficient):




    if (rebalancing_date < "2005-07-01"):
        return('调仓日期（rebalancing_date）不能早于2005年7月1日。')

    if (asset_type != 'fund' and asset_type != 'stock'):
        return('资产类型（asset_type）必须为股票或基金。')

    if (method != 'risk_parity' and method != 'min_variance' and method != 'mean_variance' and method != 'min_TE'):
        return('请选择合适的优化算法（method）。')

    if (window < 66 or type(window) != int):
        return('协方差估计样本长度（window） 必须大于66 (不少于66个交易日) ，且必须为整数。')

    if (type(cov_shrinkage) != bool):
        return('cov_shrinkage 为布尔类型变量，请选择 True 或者 False。')
        
    if (method == 'min_TE' and benchmark == 'equal_weight'):
        return('min_TE 方法需要传入指数型 benchmark。')

    if benchmark == 'equal_weight' and industry_matching == True:
        return '行业配齐需要传入指数型benchmark。'

    if method == 'mean_variance':
        if (type(expected_return) != None):
            if (type(expected_return) == pd.Series):
                missing_asset = [asset for asset in expected_return.index if asset not in  order_book_ids]
                if (len(missing_asset) != 0):
                    return('预期收益预测（expected_return）和所选合约（order_book_ids）不一致。')
                else:
                    return('预期收益预测（expected_return）的类型应为 pandas.Series。')


    #elif (expected_return_cov != None and len(expected_return_cov) != len(order_book_ids)):
    #    return('预期收益协方差矩阵（expected_return_cov）大小和资产数目（order_book_ids）不一致。')

    if (risk_aversion_coefficient < 0):
        return('风险厌恶系数（risk_aversion_coefficient）不能小于0。')


    if (asset_type == 'stock'):

        asset_list = rqdatac.instruments(order_book_ids)

        # 收集股票类资产的类型标记（场内基金，分级基金等返回的类型不是“CS”，场外基金返回 None，均不进入 list 中）
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'CS']

        if (len(asset_type_list) != len(order_book_ids)):
            return('传入的合约（order_book_ids）中包含非股票类合约。')

    if (asset_type == 'fund'):

        asset_list = rqdatac.fund.instruments(order_book_ids)

        # 收集公募基金类资产的类型标记
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'PublicFund']

        if (len(asset_type_list) != len(order_book_ids)):
            return('传入的合约（order_book_ids）中包含非基金类合约（目前仅支持公募基金）。')

    return 0

