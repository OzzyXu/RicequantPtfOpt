


###### TEST ######

import numpy as np
import rqdatac
import scipy.optimize as sc_opt
from math import *
import pandas as pd
import scipy.spatial as scsp
import sys

rqdatac.init('ricequant', '8ricequant8')

from rqdatac import *

# 传入参数

#order_book_ids = ['000001.XSHE', '000024.XSHE']
order_book_ids = ['233009', '000172']

start_date= '2004-06-01'

end_date = '2008-06-01'

asset_type = 'fund'

try:
    input_validation(order_book_ids, start_date, end_date, asset_type, method = 'risk_parity', rebalancing_frequency = 1,  window= 70.1)
except OptimizationError as e:
    print("参数输入有误：")
    print(e)


###### TEST ######



###### IMPLEMENTATION ######

class OptimizationError(Exception):

    def __init__(self, warning_message):
        print(warning_message)


def input_validation(order_book_ids, start_date, end_date, asset_type, method, rebalancing_frequency, window, bnds, cons,\
                     cov_shrinkage, expected_return, expected_return_cov, risk_aversion_coefficient, res_options):

    if (start_date < "2005-07-01"):
        raise OptimizationError('开始日期（start_date）不能早于2005年7月1日。')
        
    elif (end_date < start_date):
        raise OptimizationError('结束日期（end_date）不能早于开始日期（start_date）。')

    elif (asset_type != 'fund' and asset_type != 'stock'):
        raise OptimizationError('资产类型（asset_type）必须为股票或基金。')
    
    elif(method != 'risk_parity' and method != 'min_variance' and method != 'risk_parity_with_cons' and method != 'all'):
        raise OptimizationError('请选择合适的优化器（method）。')
        
    elif(rebalancing_frequency <= 0 or type(rebalancing_frequency) != int):
        raise OptimizationError('调仓频率（rebalancing_frequency）必须大于0，且必须为整数。')

    elif (window < 66 or type(window) != int):
        raise OptimizationError('协方差估计样本长度（window） 必须大于66 (不少于66个交易日) ，且必须为整数。')

    elif (type(cov_shrinkage) != bool):
        raise OptimizationError('cov_shrinkage 为布尔类型变量，请选择 True 或者 False。')
        
    elif (expected_return != None and len(expected_return) != len(order_book_ids)):
        raise OptimizationError('预期收益预测（expected_return）数目和资产（order_book_ids）数目不同。')
        
    elif (expected_return_cov != None and len(expected_return_cov) != len(order_book_ids)):
        raise OptimizationError('预期收益协方差矩阵（expected_return_cov）大小和资产数目（order_book_ids）不一致。')
    
    elif (risk_aversion_coefficient < 0):
        raise OptimizationError('风险厌恶系数（risk_aversion_coefficient）不能小于0。')
    
    elif (res_options != 'weights' and res_options != 'weights_indicators' and res_options != 'all'):
        raise OptimizationError('优化结果返回设置（res_options）只能选择 weights, weights_indicators 或 all。')
    
    elif (asset_type == 'stock'):
        
        asset_list = rqdatac.instruments(order_book_ids)
    
        # 收集股票类资产的类型标记（场内基金，分级基金等返回的类型不是“CS”，场外基金返回 None，均不进入 list 中）
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'CS']
        
        if (len(asset_type_list) != len(order_book_ids)):
            raise OptimizationError('传入的合约（order_book_ids）中包含非股票类合约。')
    
    elif (asset_type == 'fund'):
        
        asset_list = rqdatac.fund.instruments(order_book_ids)
    
        # 收集公募基金类资产的类型标记（场内基金，分级基金等返回的类型不是“CS”，场外基金返回 None，均不进入 list 中）
        asset_type_list = [asset.type for asset in asset_list if asset.type == 'PublicFund']
        
        if (len(asset_type_list) != len(order_book_ids)):
            raise OptimizationError('传入的合约（order_book_ids）中包含非基金类合约（目前仅支持公募基金）。')







def portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = 'all', rebalancing_frequency = 66, window= 132, bnds=None, cons=None, cov_shrinkage = True, 
                       benchmark = 'equal_weight', industry_matching = False, expected_return=None, expected_return_cov=None, risk_aversion_coefficient=1, res_options = 'weights'):
    """
    Parameters
    ----------
    order_book_ids: list 
                    A selected list of assets (stocks or funds).
    start_date: str 
                Begin date of portfolio optimization.
    end_date: str 
               End date of portfolio optimization.
    asset_type: str
                Type of assets. It can be set to be 'stock' or 'fund'.
                       
    method: str, optional, optional, default to 'all'
            Optimization algorithm, 'risk_parity', 'min_variance', 'risk_parity_with_cons', 'min_tracking_error' and 'all' are available.
             
    rebalancing_frequency: int, optional, default to 66
                           Number of trading days between every two portfolio rebalance.
                            
    bnds: dict, optional 
          Lower/upper bounds of individual asset's position in portfolio.
          (1) To impose various bounds on assets: {"asset_code1": (lb1, up1), "asset_code2": (lb2, up2), ...};
          (2) To impose identical bound on assets: {'full_list': (lb, up)}.
          
    cons: dict, optional
          Constraints on different types of asset in portfolio.
          For example, to impose allocation constraints on different asset types: {"types1": (lb1, up1), "types2": (lb2, up2), ...}.
          
    cov_shrinkage: bool, optional, default to True 
                   Set to 'True' to perform shrinkage estimation of covariane matrix.
    
    
    expected_return: 1-dimensional numpy.ndarray, optional
                     Expected returns of assets in portfolio. It is required for Mean-Variance and Black-Litterman models. 
                     Note that its dimension should be the same as 'order_book_ids'.
                     
    
    expected_return_cov: numpy.matrix optional
                         Covariance matrix of expected returns of portfolio. 
                         note that its dimension should be the same as 'order_book_ids'.
                         
    risk_aversion_coefficient: float, optional, default to 1
                               a parameter controling the importance of risk-minimation in optimization.
                               It is required for Mean-Variance and Black-Litterman models.
                               
    res_options: str, optional, default to 'weights'
                 Control which optimization results will be returned. 
                 (1) If it is set to be 'weights', only the optimized weights will be returned;
                 (2) If it is set to be 'weights_indicators', both optimized weights and performance indicators will be returned;
                 (3) If it is set to be 'all', the optimized weights, performance indicators and covariance matrix will be returned.
            
    """
    
    input_validation(order_book_ids, start_date, end_date, asset_type, method, rebalancing_frequency, window, bnds, cons,\
                     cov_shrinkage, expected_return, expected_return_cov, risk_aversion_coefficient, res_options)



    # Obtin all the trading days in the whole time period.
    # Compute the number of rebalances in the whole time period.

    trading_date_s = get_previous_trading_date(start_date)
    trading_date_e = get_previous_trading_date(end_date)
    trading_dates = get_trading_dates(trading_date_s, trading_date_e)
    time_len = len(trading_dates)
    count = floor(time_len / rebalancing_frequency)


    # Collect the rebalancing dates, and the begin/end dates of the time period.
    # The whole time period is divided into several subintervals to compute the returns of optimized portfolios.

    rebalancing_points = {}
    for i in range(0, count+1):
        rebalancing_points[i] = trading_dates[i * rebalancing_frequency]
    rebalancing_points[count+1] = trading_dates[-1]
    

    # determine the optimization algorithms.

    if method == 'all':
        methods = ['risk_parity', 'min_variance', "risk_parity_with_cons"]
    else:
        methods = method

    # using equal-weighted portfolio as benchmark

    method_keys = methods+['equal_weight']
    
    
    # Create empty series to store the arithmetic returns and portfolio values  of the optimizers.
    
    arithmetic_return_of_optimizers = {x: pd.Series() for x in method_keys}
    value_of_optimized_portfolio = {x: pd.Series() for x in method_keys}
    
    
    # loop over all time subintervals
    
    for i in range(0, count + 1):
        
        weights, cov_mat = optimizer(order_book_ids, start_date = rebalancing_points[i], asset_type, method, window, bnds, cons,\
                                     cov_shrinkage, expected_return, expected_return_cov,risk_aversion_coefficient)
                                     
        assets_list = optimized_weights.index

        if (asset_type == 'fund'):
            asset_price = fund.get_nav(assets_list, rebalancing_points[i], rebalancing_points[i+1], fields='adjusted_net_value')
            
        elif (asset_type == 'stock'):
            asset_price = rqdatac.get_price(assets_list, rebalancing_points[i], rebalancing_points[i+1], frequency='1d', fields=['close'])
 
        asset_daily_return= asset_price.pct_change()

        if i != 0:
            asset_daily_return = asset_daily_return[1:]

        # calculate portfolio arithmetic return by different methods
      
        weights[i]['equal_weight'] = [1 / len(assets_list)] * len(assets_list)
        
        for j in method_keys:
            
            # arithmetic return of portfolio
            arithmetic_return_of_portfolio = asset_daily_return.multiply(weights[i][j]).sum(axis=1)
            arithmetic_return_of_optimizers[j] = arithmetic_return_of_optimizers[j].append(arithmetic_return_of_portfolio)


            # value of optimized portfolios

            weighted_sum_of_asset_price = asset_price.multiply(weights[i][j]).sum(axis=1)
            value_of_optimized_portfolio[j] = value_of_optimized_portfolio[j].append(weighted_sum_of_asset_price)

            # Compute the indicators
            #indicators[j] = get_optimizer_indicators(weights[i][j], c_m[i], asset_type=asset_type)


    annualized_vol = {}
    annualized_cum_return = {}
    #mmd = {}
    
    for j in (method_keys):

        arithmetic_return_of_optimizers[j][0] = 0
        log_return_of_optimizers = np.log(arithmetic_return_of_optimizers[j] + 1)
        annualized_vol[j] = np.sqrt(244) * log_return_of_optimizers.std()
        days_count = len(arithmetic_return_of_optimizers[j])
        daily_cum_log_return = log_return_of_optimizers.cumsum()
        annualized_cum_return[j] = (daily_cum_log_return[-1] + 1) ** (244 / days_count) - 1

        #mmd[j] = get_maxdrawdown(daily_methods_period_price[j])


    #result_package = {'weights': weights, 'annualized_cum_return': annualized_cum_return, 'annualized_vol': annualized_vol, 'max_drawdown': max_drawdown,\
    #                  'turnover_rate': turnover_rate, 'indiviudal_asset_risk_contributions':indiviudal_asset_risk_contributions,\
    #                  'asset_class_risk_contributions': asset_class_risk_contributions, 'risk_concentration_index': risk_concentration_index, "covariance_matrix" = cov_mat }
                   
    result_package = {'weights': weights}
            
    if (res_options == 'weights'):
        res_options = ['weights']
        
    elif (res_options == 'weights_indicators'):
        res_options = ['weights', 'annualized_cum_return','annualized_vol', 'max_drawdown', 'turnover_rate', 'indiviudal_asset_risk_contributions','asset_class_risk_contributions', 'risk_concentration_index']
        
    elif (res_options == 'all'):
        res_options = ['weights', 'annualized_cum_return','annualized_vol', 'max_drawdown', 'turnover_rate', 'indiviudal_asset_risk_contributions','asset_class_risk_contributions', 'risk_concentration_index', "covariance_matrix"]
     
        
    return [result_package[x] for x in res_options]

