import numpy as np
import pandas as pd
import scipy.optimize as sc_opt
from math import *
import scipy.spatial as scsp
import sys

import rqdatac
from rqdatac import *
rqdatac.init('ricequant', '8ricequant8')

#from optimizer_for_engineer.final.input_validation import *
#from optimizer_for_engineer.final.ptfopt import *







def portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window= 132,
                       bnds=None, cons=None, cov_shrinkage = True,
                       benchmark = 'equal_weight',
                       expected_return= 'empirical_mean', risk_aversion_coef=1):

    """
    Parameters
    ----------
    order_book_ids: list
                    A selected list of assets (stocks or funds).

    rebalancing_date: str
                      date of portfolio optimization.

    asset_type: str
                Type of assets. It can be set to be 'stock' or 'fund'.

    method: str, optional, optional
            Optimization algorithm, 'risk_parity', 'min_variance', 'risk_parity_with_cons' and 'all' are available.


    bnds: dict, optional
          Lower/upper bounds of individual asset's position in portfolio.
          (1) To impose various bounds on assets: {"asset_code1": (lb1, up1), "asset_code2": (lb2, up2), ...};
          (2) To impose identical bound on assets: {'full_list': (lb, up)}.

    cons: dict, optional
          Constraints on different types of asset in portfolio.
          For example, to impose allocation constraints on different asset types: {"types1": (lb1, up1), "types2": (lb2, up2), ...}.

    cov_shrinkage: bool, optional, default to True
                   Set to 'True' to perform shrinkage estimation of covariane matrix.


    expected_return: pandas.Series, optional
                     Expected returns of assets in portfolio. It is required for Mean-Variance and Black-Litterman models.
                     Note that its dimension should be the same as 'order_book_ids'.

    risk_aversion_coefficient: float, optional, default to 1
                               a parameter controling the importance of risk-minimation in optimization.
                               It is required for Mean-Variance and Black-Litterman models.


    Results
    ----------
    total_weight: pandas.DataFrame
                  return a dataframe with order_book_ids as index, and the assets' weights and status.
                  
    optimizer_status: str
                      return a str indicating whether the optimization is successful.
                  
    """
    input_check_status = input_validation(order_book_ids, rebalancing_date, asset_type, method, window, bnds,
                     cons, cov_shrinkage, benchmark, expected_return, risk_aversion_coef)

    if input_check_status != 0:
        print(input_check_status)
        return 1
    else:
        if (cons != None or bnds != None) and method == 'risk_parity':
            method = 'risk_parity_with_con'

        try:
            weights, cov_mat, kicked_out_list, optimizer_status = optimizer(order_book_ids, rebalancing_date, asset_type, method,
                                 current_weight=None, bnds=bnds, cons=cons,
                                 expected_return=expected_return,
                                 expected_return_covar=None,
                                 risk_aversion_coefficient=risk_aversion_coef,
                                 windows=window,
                                 out_threshold_coefficient=None, data_freq=None,
                                 fun_tol=10 ** -8, max_iteration=10 ** 3, disp=False,
                                 iprint=1, cov_enhancement=cov_shrinkage, benchmark=benchmark)

        except OptimizationError:
            print(OptimizationError)
            return 1


        total_weight = pd.DataFrame(index = list(weights.index) + list(kicked_out_list.index), columns = list(weights.columns) + ['status'])
        total_weight.iloc[:,0] = weights
        total_weight.iloc[:,1] = kicked_out_list.iloc[:,0]
        total_weight = total_weight.fillna(0)

        return total_weight, optimizer_status





