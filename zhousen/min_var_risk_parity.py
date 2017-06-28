# 06/07/2017 ZS @ Ricequant


import numpy as np
import rqdatac
import scipy.optimize as sc_opt
from math import *
import pandas as pd


class OptimizationError(Exception):
    def __init__(self, warning_message):
        print(warning_message)


def data_process(order_book_ids, equity_type, start_date):

    windows = 132
    end_date = rqdatac.get_previous_trading_date(start_date)
    for i in range(windows+1):
        start_date = rqdatac.get_previous_trading_date(start_date)

    if equity_type is 'funds':
        period_data = rqdatac.fund.get_nav(order_book_ids, start_date, end_date, fields='acc_net_value')
    elif equity_type is 'stocks':
        period_data = rqdatac.get_price(order_book_ids, start_date, end_date, frequency='1d',
                                        fields=['close', 'volume'])
    period_prices = period_data['close']
    period_volume = period_data['volume']
    # Set up the threshhold of elimination
    out_threshold = ceil(period_prices.shape[0] / 2)
    kickout_list = list()
    suspended_list = list()
    # Locate the first valid value of each column, if available sequence length is less than threshhold, add
    # the column name into out_list; if sequence length is longer than threshold but less than chosen period length,
    # reset the start_date to the later date. The latest start_date whose sequence length is greater than threshold
    # will be chose.
    # Check whether any stocks has long suspended trading periods or has been delisted and generate list
    # for such stocks
    for i in order_book_ids:
        if not period_volume.loc[:, i].value_counts().empty:
            if ((end_date - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) \
                    < out_threshold:
                kickout_list.append(i)
            elif period_prices.loc[:, i].first_valid_index() < start_date:
                reset_start_date = period_prices.loc[:, i].first_valid_index()
            elif period_volume.loc[:, i].last_valid_index() < end_date or \
                            period_volume.loc[:, i].value_counts().iloc[0] >= out_threshold:
                suspended_list.append(i)
        else:
            kickout_list.append(i)
    # Check whether any ST stocks are included and generate a list for ST stocks
    st_list = list(period_prices.columns.values[rqdatac.is_st_stock(order_book_ids,
                                                                    reset_start_date, end_date).sum(axis=0) > 0])
    # Generate final kickout list which includes all the above
    final_kickout_list = list(set().union(kickout_list, st_list, suspended_list))
    # Generate clean data
    order_book_ids_s = set(order_book_ids)
    final_kickout_list_s = set(final_kickout_list)
    clean_order_book_ids = list(order_book_ids_s - final_kickout_list_s)
    clean_period_prices = period_prices.loc[reset_start_date:end_date, clean_order_book_ids]
    return clean_period_prices, final_kickout_list


def cons_func(current_weight, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = np.zeros(len(current_weight))
    if upper_bound is None:
        upper_bound = np.ones(len(current_weight))
    if len(lower_bound) != len(upper_bound):
        raise OptimizationError('Upper bound and lower bound have different size!')
    else:
        for i in range(len(current_weight)):
            if lower_bound[i] > upper_bound[i]:
                raise OptimizationError('lower bound is larger than upper bound for element %i' % i)
            else:
                bnds = bnds + [(max(0, lower_bound[i]), min(1, upper_bound[i]))]
        bnds = tuple(bnds)
    return bnds


def optimizer(order_book_ids, start_date, equity_type, method, current_weight=None, cons=None):

    if current_weight is None:
        current_weight = [1 / order_book_ids.shape[1]] * order_book_ids.shape[1]

    data_after_processing = data_process(order_book_ids, equity_type, start_date)
    clean_period_prices = data_after_processing[0]
    period_daily_return_pct_change = clean_period_prices.pct_change()
    c_m = period_daily_return_pct_change.cov()

    # Log barrier risk parity model
    c = 15
    log_barrier_risk_parity_obj_fun = lambda x: np.dot(np.dot(x, c_m), x) - c * sum(np.log(x))
    log_barrier_bnds = []
    for i in range(len(current_weight)):
        log_barrier_bnds = log_barrier_bnds + [(0.00001, 1)]
    log_barrier_bnds = tuple(log_barrier_bnds)

    def log_barrier_risk_parity_gradient(x):
        return np.multiply(2, np.dot(c_m, x)) - np.multiply(c, np.reciprocal(x))

    def log_barrier_risk_parity_optimizer():
        if cons is None:
            optimization_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, current_weight, method='L-BFGS-B',
                                               jac=log_barrier_risk_parity_gradient, bounds=log_barrier_bnds)
        else:
            optimization_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, current_weight, method='SLSQP',
                                               jac=log_barrier_risk_parity_gradient, constraints=cons)
        if not optimization_res.success:
            temp = ' @ %s' % clean_period_prices.index[0]
            error_message = 'Risk parity optimization failed, ' + str(optimization_res.message) + temp
            raise OptimizationError(error_message)
        else:
            optimal_weights = (optimization_res.x / sum(optimization_res.x))
            return optimal_weights

    # Min variance model
    min_variance_obj_fun = lambda x: np.dot(np.dot(x, c_m), x)
    min_variance_cons_fun = lambda x: sum(x) - 1
    min_variance_default_cons = ({'type': 'eq', 'fun': min_variance_cons_fun})
    min_variance_bnds = []
    for i in range(len(current_weight)):
        min_variance_bnds = min_variance_bnds + [(0, 1)]
    min_variance_bnds = tuple(min_variance_bnds)

    def min_variance_gradient(x):
        return np.multiply(2, np.dot(c_m, x))

    def min_variance_optimizer():
        if cons is None:
            optimization_res = sc_opt.minimize(min_variance_obj_fun, current_weight, method='SLSQP',
                                               jac=min_variance_gradient, bounds=min_variance_bnds,
                                               constraints=min_variance_default_cons)
        else:
            optimization_res = sc_opt.minimize(min_variance_obj_fun, current_weight, method='SLSQP',
                                               jac=min_variance_gradient, constraints=cons)
        if not optimization_res.success:
            temp = ' @ %s' % clean_period_prices.index[0]
            error_message = 'Min variance optimization failed, ' + str(optimization_res.message) + temp
            raise OptimizationError(error_message)
        else:
            return optimization_res.x

    opt_dict = {'risk_parity': log_barrier_risk_parity_optimizer,
                'min_variance': min_variance_optimizer,
                'all': [log_barrier_risk_parity_optimizer, min_variance_optimizer]}

    if method is not 'all':
        return pd.DataFrame(opt_dict[method](), index=list(clean_period_prices.columns.value)), data_after_processing[1]
    else:
        temp1 = list()
        for f in opt_dict[method]:
            temp1.append(pd.DataFrame(f(), index=list(clean_period_prices.columns.value)))
        return temp1, data_after_processing[1]