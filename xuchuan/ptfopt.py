# 06/07/2017 By Chuan Xu @ Ricequant V 3.0


import numpy as np
import rqdatac
import scipy.optimize as sc_opt
from math import *
import pandas as pd

rqdatac.init('ricequant', '8ricequant8')

class OptimizationError(Exception):

    def __init__(self, warning_message):
        print(warning_message)


# Clean data for covariance matrix calculation
def data_process(order_book_ids, equity_type, start_date):

    windows = 132
    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    for i in range(windows + 1):
        start_date = rqdatac.get_previous_trading_date(start_date)
    start_date = pd.to_datetime(start_date)
    reset_start_date = start_date

    if equity_type is 'fund':
        period_prices = rqdatac.fund.get_nav(order_book_ids, reset_start_date, end_date, fields='acc_net_value')
    elif equity_type is 'stock':
        period_data = rqdatac.get_price(order_book_ids, reset_start_date, end_date, frequency='1d',
                                        fields=['close', 'volume'])
        period_prices = period_data['close']
        period_volume = period_data['volume']
    # Set up the threshhold of elimination
    out_threshold = ceil(period_prices.shape[0] / 2)
    kickout_list = list()
    suspended_list = list()
    st_list = list()
    # Locate the first valid value of each column, if available sequence length is less than threshhold, add
    # the column name into out_list; if sequence length is longer than threshold but less than chosen period length,
    # reset the start_date to the later date. The latest start_date whose sequence length is greater than threshold
    # will be chose.
    # Check whether any stocks has long suspended trading periods or has been delisted and generate list
    # for such stocks
    if equity_type is "stock":
        for i in order_book_ids:
            if not period_volume.loc[:, i].value_counts().empty:
                if ((end_date - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) \
                        < out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].first_valid_index() < reset_start_date:
                    reset_start_date = period_prices.loc[:, i].first_valid_index()
                elif period_volume.loc[:, i].last_valid_index() < end_date or \
                                period_volume.loc[:, i].value_counts().iloc[0] >= out_threshold:
                    suspended_list.append(i)
            else:
                kickout_list.append(i)
        # Check whether any ST stocks are included and generate a list for ST stocks
        st_list = list(period_prices.columns.values[rqdatac.is_st_stock(order_book_ids,
                                                                        reset_start_date, end_date).sum(axis=0) > 0])
    elif equity_type is "fund":
        for i in order_book_ids:
            if period_prices.loc[:, i].first_valid_index() is not None:
                if ((end_date - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) < out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].first_valid_index() < reset_start_date:
                    reset_start_date = period_prices.loc[:, i].first_valid_index()
            else:
                kickout_list.append(i)
    # Generate final kickout list which includes all the above
    final_kickout_list = list(set().union(kickout_list, st_list, suspended_list))
    # Generate clean data
    order_book_ids_s = set(order_book_ids)
    final_kickout_list_s = set(final_kickout_list)
    clean_order_book_ids = list(order_book_ids_s - final_kickout_list_s)
    clean_period_prices = period_prices.loc[reset_start_date:end_date, clean_order_book_ids]
    return clean_period_prices, final_kickout_list


# Generate upper and lower bounds for equities in portfolio
def bounds_gen(order_book_ids, clean_order_book_ids, method, bounds=None):

    if bounds is not None:
        for key in bounds:
            if key is not "full_list" and key not in order_book_ids:
                raise OptimizationError('Bounds contain equities not existing in pool! ')
            elif bounds[key][0] > bounds[key][1]:
                raise OptimizationError("Lower bound is larger than upper bound for some equities!")

        general_bnds = list()
        log_rp_bnds = list()
        if method is "risk_parity":
            log_rp_bnds = [(10**-6, inf)] * len(clean_order_book_ids)
        elif method is "all":
            log_rp_bnds = [(10 ** -6, inf)] * len(clean_order_book_ids)
            for i in clean_order_book_ids:
                if "full_list" in list(bounds):
                    general_bnds = general_bnds + [(max(0, bounds["full_list"][0]), min(1, bounds["full_list"][1]))]
                elif i in list(bounds):
                    general_bnds = general_bnds + [(max(0, bounds[i][0]), min(1, bounds[i][1]))]
                else:
                    general_bnds = general_bnds + [(0, 1)]
        else:
            for i in clean_order_book_ids:
                if "full_list" in list(bounds):
                    general_bnds = general_bnds + [(max(0, bounds["full_list"][0]), min(1, bounds["full_list"][1]))]
                elif i in list(bounds):
                    general_bnds = general_bnds + [(max(0, bounds[i][0]), min(1, bounds[i][1]))]
                else:
                    general_bnds = general_bnds + [(0, 1)]

        if method is "all":
            return tuple(log_rp_bnds), tuple(general_bnds)
        elif method is "risk_parity":
            return tuple(log_rp_bnds)
        else:
            return tuple(general_bnds)
    else:
        log_rp_bnds = [(10**-6, inf)] * len(clean_order_book_ids)
        general_bnds = [(0, 1)] * len(clean_order_book_ids)
        if method is "all":
            return tuple(log_rp_bnds), tuple(general_bnds)
        elif method is "risk_parity":
            return tuple(log_rp_bnds)
        else:
            return tuple(general_bnds)


# Generate category constraints for portfolio
def constraints_gen(clean_order_book_ids, equity_type, method, constraints=None):

    if constraints is not None:
        df = pd.DataFrame(index=clean_order_book_ids, columns=['type'])

        for key in constraints:
            if constraints[key][0] > constraints[key][1]:
                raise OptimizationError("Constraints setup error!")

        if equity_type is 'fund':
            for i in clean_order_book_ids:
                df.loc[i, 'type'] = rqdatac.fund.instruments(i).fund_type
        elif equity_type is 'stock':
            for i in clean_order_book_ids:
                df.loc[i, "type"] = rqdatac.instruments(i).shenwan_industry_name

        cons = list()
        for key in constraints:
            key_list = list(df[df['type'] == key].index)
            key_pos_list = list()
            for i in key_list:
                key_pos_list.append(df.index.get_loc(i))
            key_cons_fun_lb = lambda x: sum(x[t] for t in key_pos_list) - constraints[key][0]
            key_cons_fun_ub = lambda x: constraints[key][1] - sum(x[t] for t in key_pos_list)
            cons.append({"type": "ineq", "fun": key_cons_fun_lb})
            cons.append({"type": "ineq", "fun": key_cons_fun_ub})

        return tuple(cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1}))
    else:
        return {'type': 'eq', 'fun': lambda x: sum(x) - 1}


# order_book_ids: list. A list of equities(stocks or funds)
# start_date: str. Date to set up portfolio or rebalance portfolio
# equity_type: str or str list. Types of portfolio candidates,  "stock" or "fund", portfolio with mixed equities
#              is not supportted;
# method: str. Portfolio optimization model: "risk_parity", "min_variance", "all";
# current_weight: float list. Candidates' weights of current portfolio. Will be set as the initial guess
#                 to start optimization. None type input means equal weights portfolio.
# bnds: float list. Lower bounds and upper bounds for each equity in portfolio. Support input format: {equity_code1: (lb1, up1),
# equity_code2: (lb2, up2), ...} or {'full_list': (lb, up)}(set up universal bounds for all);
# cons: dict. Lower bounds and upper bounds for each category of equities in portfolio.
# Funds type: Bond, Stock, Hybrid, Money, ShortBond, StockIndex, BondIndex, Related, QDII, Other;
# Stocks type: Shenwan_industry_name
# Support input format: {types1: (lb1, up1), types2: (lb2,up2), ...}
def optimizer(order_book_ids, start_date, equity_type, method, current_weight=None, bnds=None, cons=None):

    data_after_processing = data_process(order_book_ids, equity_type, start_date)
    clean_period_prices = data_after_processing[0]
    period_daily_return_pct_change = clean_period_prices.pct_change()
    c_m = period_daily_return_pct_change.cov()

    if current_weight is None:
        current_weight = [1 / clean_period_prices.shape[1]] * clean_period_prices.shape[1]
    else:
        new_current_weight = current_weight
        current_weight = list()
        for i in clean_period_prices.columns.values:
            current_weight.append(new_current_weight[order_book_ids.index(i)])

    if method is "all":
        log_rp_bnds, general_bnds = bounds_gen(order_book_ids, list(clean_period_prices.columns), method, bnds)
    elif method is "risk_parity":
        log_rp_bnds = bounds_gen(order_book_ids, list(clean_period_prices.columns), method, bnds)
    else:
        general_bnds = bounds_gen(order_book_ids, list(clean_period_prices.columns), method, bnds)
    general_cons = constraints_gen(list(clean_period_prices.columns), equity_type, method, cons)

    # Log barrier risk parity modek
    c = 15

    def log_barrier_risk_parity_obj_fun(x):
        return np.dot(np.dot(x, c_m), x) - c * sum(np.log(x))

    def log_barrier_risk_parity_gradient(x):
        return np.multiply(2, np.dot(c_m, x)) - np.multiply(c, np.reciprocal(x))

    def log_barrier_risk_parity_optimizer():
        optimization_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, current_weight, method='L-BFGS-B',
                                           jac=log_barrier_risk_parity_gradient, bounds=log_rp_bnds)

        if not optimization_res.success:
            temp = ' @ %s' % clean_period_prices.index[0]
            error_message = 'Risk parity optimization failed, ' + str(optimization_res.message) + temp
            raise OptimizationError(error_message)
        else:
            optimal_weights = (optimization_res.x / sum(optimization_res.x))
            return optimal_weights

    # Min variance model
    min_variance_obj_fun = lambda x: np.dot(np.dot(x, c_m), x)

    def min_variance_gradient(x):
        return np.multiply(2, np.dot(c_m, x))

    def min_variance_optimizer():
        optimization_res = sc_opt.minimize(min_variance_obj_fun, current_weight, method='SLSQP',
                                           jac=min_variance_gradient, bounds=general_bnds, constraints=general_cons,
                                           options={"ftol": 10**-12})

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
        return pd.DataFrame(opt_dict[method](), index=clean_period_prices.columns.values, columns=[method]), \
               data_after_processing[1]
    else:
        temp1 = pd.DataFrame(index=clean_period_prices.columns.values, columns=['risk_parity', 'min_variance'])
        n = 0
        for f in opt_dict[method]:
            temp1.iloc[:, n] = f()
            n = n + 1
        return temp1, data_after_processing[1]
