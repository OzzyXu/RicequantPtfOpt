import rqdatac
from rqdatac import *
rqdatac.init('ricequant', '8ricequant8')


class OptimizationError(Exception):
    def __init__(self, warning_message):
        print(warning_message)


def input_validation(order_book_ids, start_date, end_date, asset_type, method, rebalancing_frequency, window, bnds,
                     cons, \
                     cov_shrinkage, expected_return,  risk_aversion_coefficient, res_options):
    if (start_date < "2005-07-01"):
        # return('开始日期（start_date）不能早于2005年7月1日。')
        return('开始日期（start_date）不能早于2005年7月1日。')

    elif (end_date < start_date):
        return('结束日期（end_date）不能早于开始日期（start_date）。')

    elif (asset_type != 'fund' and asset_type != 'stock'):
        return('资产类型（asset_type）必须为股票或基金。')

    elif (method != 'risk_parity' and method != 'min_variance' and method != 'risk_parity_with_cons' and method != 'all'):
        return('请选择合适的优化器（method）。')

    elif (rebalancing_frequency <= 0 or type(rebalancing_frequency) != int):
        return('调仓频率（rebalancing_frequency）必须大于0，且必须为整数。')

    elif (window < 66 or type(window) != int):
        return('协方差估计样本长度（window） 必须大于66 (不少于66个交易日) ，且必须为整数。')

    elif (type(cov_shrinkage) != bool):
        return('cov_shrinkage 为布尔类型变量，请选择 True 或者 False。')
        
    elif (method == 'min_TE' and benchmark == 'equal_weight'):
        return('min_TE 方法需要传入指数型benchmark。')

    #elif (expected_return != 'empirical_mean' and len(expected_return) != len(order_book_ids)):
    #   return('预期收益预测（expected_return）数目和资产（order_book_ids）数目不同。')

    #elif (expected_return_cov != None and len(expected_return_cov) != len(order_book_ids)):
    #   return('预期收益协方差矩阵（expected_return_cov）大小和资产数目（order_book_ids）不一致。')

    elif (risk_aversion_coefficient < 0):
        return('风险厌恶系数（risk_aversion_coefficient）不能小于0。')

    elif (res_options != 'weight' and res_options != 'weight_indicators' and res_options != 'all'):
        return('优化结果返回设置（res_options）只能选择 weight, weight_indicators 或 all。')

    elif (asset_type == 'stock'):

        asset_list = rqdatac.instruments(order_book_ids)

        # 收集股票类资产的类型标记（场内基金，分级基金等返回的类型不是“CS”，场外基金返回 None，均不进入 list 中）
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'CS']

        if (len(asset_type_list) != len(order_book_ids)):
            return('传入的合约（order_book_ids）中包含非股票类合约。')
        else:
            return(0)

    elif (asset_type == 'fund'):

        asset_list = rqdatac.fund.instruments(order_book_ids)

        # 收集公募基金类资产的类型标记
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'PublicFund']

        if (len(asset_type_list) != len(order_book_ids)):
            return('传入的合约（order_book_ids）中包含非基金类合约（目前仅支持公募基金）。')
        else:
            return(0)
    else:
        return(0)

