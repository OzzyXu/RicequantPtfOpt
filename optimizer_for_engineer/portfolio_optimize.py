import numpy as np
import pandas as pd
import scipy.optimize as sc_opt
from math import *
import scipy.spatial as scsp
import sys

import rqdatac
from rqdatac import *
rqdatac.init('ricequant', '8ricequant8')



class OptimizationError(Exception):
    def __init__(self, warning_message):
        print(warning_message)




def portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = 'all',
                       rebalancing_frequency = 66, window= 132, bnds=None, cons=None,
                       cov_shrinkage = True, benchmark = 'equal_weight',
                       industry_matching = False, expected_return= 'empirical_mean',
                       risk_aversion_coef=1, res_options = 'weight'):

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
            Optimization algorithm, 'risk_parity', 'min_variance', 'mean_variance', 'min_TE' and 'all' are available.

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


    expected_return: pandas.series, optional
                     Expected returns of assets in portfolio. It is required for Mean-Variance and Black-Litterman models.
                     Note that its dimension should be the same as 'order_book_ids'.


    expected_return_cov: pandas.dataframe optional
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
    input_check_status = input_validation(order_book_ids, start_date, end_date, asset_type, method, rebalancing_frequency, window, bnds,
                     cons, cov_shrinkage, expected_return, risk_aversion_coef, res_options)

    if input_check_status != 0:
        print(input_check_status)
    else:
        # Obtain all the trading days in the whole time period.
        # Compute the number of rebalances in the whole time period.

        trading_date_s = get_previous_trading_date(start_date)
        trading_date_e = get_previous_trading_date(end_date)
        trading_dates = get_trading_dates(trading_date_s, trading_date_e)
        time_len = len(trading_dates)
        count = floor(time_len / rebalancing_frequency)

        # Collect the rebalancing dates, and the begin/end dates of the time period.
        # The whole time period is divided into several subintervals to compute the returns of optimized portfolios.

        rebalancing_points = {}
        for i in range(0, count + 1):
            rebalancing_points[i] = trading_dates[i * rebalancing_frequency]
        rebalancing_points[count + 1] = trading_dates[-1]

        # determine the optimization algorithms.

        if method == 'all':
            methods = ['risk_parity', 'min_variance', "risk_parity_with_cons"]
        
        elif (method == 'risk_parity' and bnds != None) or (method == 'risk_parity' and cons != None):
            methods = "risk_parity_with_cons"
            
        else:
            methods = method

        # using equal-weighted portfolio as benchmark

        method_keys = methods + ['equal_weight']

        # Create empty series to store the arithmetic returns and portfolio values  of the optimizers.

        arithmetic_return_of_optimizers = {x: pd.Series() for x in method_keys}
        #value_of_optimized_portfolio = {x: pd.Series() for x in method_keys}

        # loop over all time subintervals

        for i in range(0, count + 1):


            weights, cov_mat, kicked_out_list = optimizer(order_book_ids, start_date, asset_type, method,
                                                          current_weight=None, bnds=bnds, cons=cons,
                                                          expected_return=expected_return,
                                                          expected_return_covar=None,
                                                          risk_aversion_coefficient=risk_aversion_coef,
                                                          windows=window,
                                                          out_threshold_coefficient=None, data_freq=None,
                                                          fun_tol=10 ** -8, max_iteration=10 ** 5, disp=False,
                                                          iprint=1, cov_enhancement=cov_shrinkage, benchmark=benchmark)


            assets_list = list(weights.index)

            if (asset_type == 'fund'):
                asset_price = fund.get_nav(assets_list, rebalancing_points[i], rebalancing_points[i + 1],
                                           fields='adjusted_net_value')

            elif (asset_type == 'stock'):
                asset_price = rqdatac.get_price(assets_list, rebalancing_points[i], rebalancing_points[i + 1],
                                                frequency='1d', fields=['close'])

            asset_daily_return = asset_price.pct_change()

            if i != 0:
                asset_daily_return = asset_daily_return[1:]

            # calculate portfolio arithmetic return by different methods

            weights[i]['equal_weight'] = [1 / len(assets_list)] * len(assets_list)

            for j in method_keys:
                # arithmetic return of portfolio
                arithmetic_return_of_portfolio = asset_daily_return.multiply(weights[i][j]).sum(axis=1)
                arithmetic_return_of_optimizers[j] = arithmetic_return_of_optimizers[j].append(
                    arithmetic_return_of_portfolio)

                # value of optimized portfolios

                # weighted_sum_of_asset_price = asset_price.multiply(weights[i][j]).sum(axis=1)
                # value_of_optimized_portfolio[j] = value_of_optimized_portfolio[j].append(weighted_sum_of_asset_price)

                # Compute the indicators
                # indicators[j] = get_optimizer_indicators(weights[i][j], c_m[i], asset_type=asset_type)

        annualized_vol = {}
        annualized_cum_return = {}
        # mmd = {}

        for j in (method_keys):
            arithmetic_return_of_optimizers[j][0] = 0
            log_return_of_optimizers = np.log(arithmetic_return_of_optimizers[j] + 1)
            annualized_vol[j] = np.sqrt(244) * log_return_of_optimizers.std()
            days_count = len(arithmetic_return_of_optimizers[j]) + 1
            daily_cum_log_return = log_return_of_optimizers.cumsum()
            annualized_cum_return[j] = (daily_cum_log_return[-1] + 1) ** (244 / days_count) - 1

            # mmd[j] = get_maxdrawdown(daily_methods_period_price[j])

        # result_package = {'weights': weights, 'annualized_cum_return': annualized_cum_return, 'annualized_vol': annualized_vol, 'max_drawdown': max_drawdown,\
        #                  'turnover_rate': turnover_rate, 'indiviudal_asset_risk_contributions':indiviudal_asset_risk_contributions,\
        #                  'asset_class_risk_contributions': asset_class_risk_contributions, 'risk_concentration_index': risk_concentration_index, "covariance_matrix" = cov_mat }

        result_package = {'weights': weights}

        if (res_options == 'weights'):
            res_options = ['weights']

        elif (res_options == 'weights_indicators'):
            res_options = ['weights', 'annualized_cum_return', 'annualized_vol', 'max_drawdown', 'turnover_rate',
                           'indiviudal_asset_risk_contributions', 'asset_class_risk_contributions',
                           'risk_concentration_index']

        elif (res_options == 'all'):
            res_options = ['weights', 'annualized_cum_return', 'annualized_vol', 'max_drawdown', 'turnover_rate',
                           'indiviudal_asset_risk_contributions', 'asset_class_risk_contributions',
                           'risk_concentration_index', "covariance_matrix"]

        return [result_package[x] for x in res_options]

