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


def data_process(order_book_ids, asset_type, start_date, windows):

    """
    Clean data for covariance matrix calculation
    :param order_book_ids: str list. A group of assets.
    :param asset_type: str. "fund" or "stock"
    :param start_date: str. The first day of backtest period
    :param windows: int. Interval length of sample
    :return: DataFrame, str list, str. The DataFrame contains the prices after cleaning; the str list contains the
             order_book_ids been filtered out due to unqualified in covariance matrix calculation; the new start date
             of covariance calculation interval which may differ from default.
    """

    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = rqdatac.get_trading_dates("2005-01-01", end_date)[-windows-1]
    reset_start_date = pd.to_datetime(start_date)

    if asset_type is 'fund':
        period_prices = rqdatac.fund.get_nav(order_book_ids, reset_start_date, end_date, fields='adjusted_net_value')
    elif asset_type is 'stock':
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
    if asset_type is "stock":
        for i in order_book_ids:
            if not period_volume.loc[:, i].value_counts().empty:
                if ((end_date - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) \
                        < out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].isnull().sum() >= out_threshold:
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
    elif asset_type is "fund":
        for i in order_book_ids:
            if period_prices.loc[:, i].first_valid_index() is not None:
                if ((end_date - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) < out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].isnull().sum() >= out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].first_valid_index() < reset_start_date:
                    reset_start_date = period_prices.loc[:, i].first_valid_index()
            else:
                kickout_list.append(i)
    period_prices = period_prices.fillna(method="pad")
    # Generate final kickout list which includes all the above
    final_kickout_list = list(set().union(kickout_list, st_list, suspended_list))
    # Generate clean data
    final_kickout_list_s = set(final_kickout_list)
    # Keep the original input id order
    clean_order_book_ids = [x for x in order_book_ids if x not in final_kickout_list_s]

    clean_period_prices = period_prices.loc[reset_start_date:end_date, clean_order_book_ids]
    return clean_period_prices, final_kickout_list, reset_start_date


def black_litterman_prep(order_book_ids, start_date, investors_views, investors_views_indicate_M,
                         investors_views_uncertainty=None, asset_type=None, market_weight=None,
                         risk_free_rate_tenor=None, risk_aversion_coefficient=None, excess_return_cov_uncertainty=None,
                         confidence_of_views=None, windows=None):
    """
    Generate expected return and expected return covariance matrix with Black-Litterman model. Suppose we have N assets
    and K views.
    :param order_book_ids: str list. A group of assets;
    :param asset_type: str. "fund" or "stock";
    :param start_date: str. The first day of backtest period;
    :param windows: int. Interval length of sample;
    :param investors_views: K*1 numpy matrix. Each row represents one view;
    :param investors_views_indicate_M: K*N numpy matrix. Each row corresponds to one view. Indicate which view is
    involved during calculation;
    :param investors_views_uncertainty: K*K diagonal matrix, optional. If it is skipped, He and Litterman's method will
    be called to generate diagonal matrix if confidence_of_view is also skipped; Idzorek's method will be called if
    confidence_of_view is passed in; Has to be non-singular;
    :param market_weight: floats list, optional. Weights for market portfolio; Default: Equal weights portfolio;
    :param risk_free_rate_tenor: str, optional. The period of risk free rate will be used. Default: "0s";
    :param risk_aversion_coefficient: float, optional. If no risk_aversion_coefficient is passed in, then
    risk_aversion_coefficient = market portfolio risk premium / market portfolio volatility;
    :param excess_return_cov_uncertainty: float, optional. Default: 1/T where T is the time length of sample;
    :param confidence_of_views: floats list, optional. Represent investors' confidence levels on each view.
    :return: expected return vector, covariance matrix of expected return, risk_aversion_coefficient,
    investors_views_uncertainty.
    ########
    # It's highly recommended to use your own ways to create investors_views_uncertainty, risk_aversion_coefficient and
    # excess_return_cov_uncertainty beforehand to get the desired distribution parameters.
    ########
    """

    risk_free_rate_dict = ['0S', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y',
                           '9Y', '10Y', '15Y', '20Y', '30Y', '40Y', '50Y']

    if market_weight is None:
        market_weight = pd.DataFrame([1 / len(order_book_ids)] * len(order_book_ids), index=order_book_ids)
    if windows is None:
        windows = 132

    # Clean data
    if asset_type is None:
        asset_type = "fund"
    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    clean_period_prices, reset_start_date = (data_process(order_book_ids, asset_type, start_date, windows)[i]
                                             for i in [0, 2])

    if excess_return_cov_uncertainty is None:
        excess_return_cov_uncertainty = 1 / clean_period_prices.shape[0]

    reset_start_date = rqdatac.get_next_trading_date(reset_start_date)
    # Take daily risk free rate
    if risk_free_rate_tenor is None:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor='0S', country='cn')
    elif risk_free_rate_tenor in risk_free_rate_dict:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor=risk_free_rate_tenor,
                                                 country='cn')
    risk_free_rate['Daily'] = pd.Series(np.power(1 + risk_free_rate['0S'], 1 / 365) - 1, index=risk_free_rate.index)

    # Calculate daily risk premium for each equity
    clean_period_prices_pct_change = clean_period_prices.pct_change()
    clean_period_excess_return = clean_period_prices_pct_change.subtract(risk_free_rate['Daily'], axis=0)

    # Wash out the ones in kick_out_list
    clean_market_weight = market_weight.loc[clean_period_prices.columns.values]
    temp_sum_weight = clean_market_weight.sum()
    clean_market_weight = clean_market_weight.div(temp_sum_weight)

    # If no risk_aversion_coefficient is passed in, then
    # risk_aversion_coefficient = market portfolio risk premium / market portfolio volatility
    if risk_aversion_coefficient is None:
        market_portfolio_return = np.dot(clean_period_prices_pct_change, clean_market_weight)
        risk_aversion_coefficient = ((market_portfolio_return[1:].mean() - risk_free_rate["Daily"].mean()) /
                                     market_portfolio_return[1:].var())

    equilibrium_return = np.multiply(np.dot(clean_period_excess_return[1:].cov(), clean_market_weight),
                                     risk_aversion_coefficient)

    clean_period_excess_return_cov = clean_period_excess_return[1:].cov()
    # Generate the investors_views_uncertainty matrix if none is passed in
    if investors_views_uncertainty is None:
        if confidence_of_views is None:
            # He and Litteman's(1999) method to generate the uncertainty diagonal matrix, confidence level on each view
            # doesn't need.
            Omeg_diag = list()
            for i in range(investors_views_indicate_M.shape[0]):
                temp = np.dot(np.dot(investors_views_indicate_M[i, :], clean_period_excess_return_cov),
                              investors_views_indicate_M[i, :].transpose()) * excess_return_cov_uncertainty
                Omeg_diag.append(temp.item(0))
            investors_views_uncertainty = np.diag(Omeg_diag)
        else:
            # Idzorek's(2002) method, users can specify their confidence level on each view.
            Omeg_diag = list()
            for i in range(len(investors_views)):
                part1 = excess_return_cov_uncertainty * np.dot(clean_period_excess_return_cov,
                                                               investors_views_indicate_M[i, :].transpose())
                part2 = 1 / (excess_return_cov_uncertainty * np.dot(investors_views_indicate_M[i, :],
                                                                    np.dot(clean_period_excess_return_cov,
                                                                           investors_views_indicate_M[i,
                                                                           :].transpose())))
                part3 = investors_views[i] - np.dot(investors_views_indicate_M[i, :], equilibrium_return)
                return_with_full_confidence = equilibrium_return + np.multiply(part2 * part3, part1)
                weights_with_full_confidence = np.dot(np.linalg.inv(np.multiply(risk_aversion_coefficient,
                                                                                clean_period_excess_return_cov)),
                                                      return_with_full_confidence)
                temp1 = weights_with_full_confidence - clean_market_weight
                temp2 = np.multiply(confidence_of_views[i], np.absolute(investors_views_indicate_M[i, :].transpose()))
                tilt = np.multiply(temp1, temp2)
                weights_with_partial_confidence = clean_market_weight.as_matrix() + tilt

                def objective_fun(x):
                    temp1 = np.linalg.inv(np.multiply(risk_aversion_coefficient, clean_period_excess_return_cov))
                    temp2 = np.linalg.inv(np.linalg.inv(np.multiply(excess_return_cov_uncertainty,
                                                                    clean_period_excess_return_cov)) +
                                          np.multiply(np.reciprocal(x),
                                                      np.dot(investors_views_indicate_M[i, :].transpose(),
                                                             investors_views_indicate_M[i, :])))
                    temp3 = (np.dot(np.linalg.inv(np.multiply(excess_return_cov_uncertainty,
                                                              clean_period_excess_return_cov)), equilibrium_return) +
                             np.multiply(investors_views[i] * np.reciprocal(x),
                                         investors_views_indicate_M[i, :].transpose()))
                    wk = np.dot(temp1, np.dot(temp2, temp3))
                    return np.linalg.norm(np.subtract(weights_with_partial_confidence, wk))

                # Upper bound should be consistent with the magnitude of return
                upper_bound = abs(equilibrium_return.mean()) * 100
                omega_k = sc_opt.minimize_scalar(objective_fun, bounds=(10 ** -8, upper_bound), method="bounded",
                                                 options={"xatol": 10 ** -8})
                Omeg_diag.append(omega_k.x.item(0))
            investors_views_uncertainty = np.diag(Omeg_diag)

    # Combine all the information above to get the distribution of expected return with given views
    combined_return_covar = np.linalg.inv(np.linalg.inv(np.multiply(excess_return_cov_uncertainty,
                                                                    clean_period_excess_return_cov))
                                          + np.dot(np.dot(investors_views_indicate_M.transpose(),
                                                          np.linalg.inv(investors_views_uncertainty)),
                                                   investors_views_indicate_M))
    temp1 = np.dot(np.linalg.inv(np.multiply(excess_return_cov_uncertainty, clean_period_excess_return_cov)),
                   equilibrium_return)
    temp2 = np.dot(np.dot(investors_views_indicate_M.transpose(), np.linalg.inv(investors_views_uncertainty)),
                   investors_views)
    temp = temp1 + temp2

    combined_return_mean = np.dot(combined_return_covar, temp)

    return combined_return_mean, combined_return_covar, risk_aversion_coefficient, investors_views_uncertainty


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
def constraints_gen(clean_order_book_ids, asset_type, constraints=None):

    if constraints is not None:
        df = pd.DataFrame(index=clean_order_book_ids, columns=['type'])

        for key in constraints:
            if constraints[key][0] > constraints[key][1]:
                raise OptimizationError("Constraints setup error!")

        if asset_type is 'fund':
            for i in clean_order_book_ids:
                df.loc[i, 'type'] = rqdatac.fund.instruments(i).fund_type
        elif asset_type is 'stock':
            for i in clean_order_book_ids:
                df.loc[i, "type"] = rqdatac.instruments(i).shenwan_industry_name

        cons = list()
        for key in constraints:
            if key not in df.type.unique():
                raise OptimizationError("Non-existing category in constraints: %s" % key)
            key_list = list(df[df['type'] == key].index)
            key_pos_list = list()
            for i in key_list:
                key_pos_list.append(df.index.get_loc(i))
            key_cons_fun_lb = lambda x: sum(x[t] for t in key_pos_list) - constraints[key][0]
            key_cons_fun_ub = lambda x: constraints[key][1] - sum(x[t] for t in key_pos_list)
            cons.append({"type": "ineq", "fun": key_cons_fun_lb})
            cons.append({"type": "ineq", "fun": key_cons_fun_ub})
        cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        return tuple(cons)
    else:
        return {'type': 'eq', 'fun': lambda x: sum(x) - 1}


# order_book_ids: list. A list of assets(stocks or funds);
# start_date: str. Date to initialize a portfolio or rebalance a portfolio;
# asset_type: str or str list. Types of portfolio candidates,  "stock" or "fund", portfolio with mixed equities
#              is not supportted;
# method: str. Portfolio optimization model: "risk_parity", "min_variance", "mean_variance", "all";
# current_weight: list of floats, optional. Candidates' weights of current portfolio. Will be used as the initial guess
#                 to start optimization. Default: equal weights portfolio;
# bnds: list of floats. Lower bounds and upper bounds for each equity in portfolio.
#       Support input format: {equity_code1: (lb1, up1), equity_code2: (lb2, up2), ...} or {'full_list': (lb, up)}
#       (set up universal bounds for all);
# cons: dict, optional. Lower bounds and upper bounds for each category of equities in portfolio;
# Supported funds type: Bond, Stock, Hybrid, Money, ShortBond, StockIndex, BondIndex, Related, QDII, Other; supported
# stocks type: Shenwan_industry_name;
# cons: {types1: (lb1, up1), types2: (lb2, up2), ...};
# expected_return: column vector of floats, optional. Default: Means of the returns of order_book_ids within windows.
# expected_return_covar: numpy matrix, optional. Covariance matrix of expected return. Default: covariance of the means
#                        of the returns of order_book_ids within windows;
# risk_aversion_coefficient: float, optional. Risk aversion coefficient of Mean-Variance model. Default: 1.
def optimizer(order_book_ids, start_date, asset_type, method, current_weight=None, bnds=None, cons=None,
              expected_return=None, expected_return_covar=None, risk_aversion_coefficient=1):

    # Get clean data and calculate covariance matrix
    windows = 132
    data_after_processing = data_process(order_book_ids, asset_type, start_date, windows)
    clean_period_prices = data_after_processing[0]
    period_daily_return_pct_change = clean_period_prices.pct_change()
    c_m = period_daily_return_pct_change.cov()

    if clean_period_prices.shape[1] == 0:
        # print('All selected funds have been ruled out')
        return  data_after_processing[1]
    else:

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
        general_cons = constraints_gen(list(clean_period_prices.columns), asset_type, cons)

        # Log barrier risk parity model
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

        # Mean variance model
        if expected_return is None:
            expected_return = period_daily_return_pct_change.mean()
        if expected_return_covar is None:
            expected_return_covar = c_m

        def mean_variance_obj_fun(x):
            return (np.multiply(risk_aversion_coefficient/2, np.dot(np.dot(x, expected_return_covar), x)) -
                    np.dot(x, expected_return))

        def mean_variance_gradient(x):
            return np.asfarray(np.multiply(risk_aversion_coefficient, np.dot(x, expected_return_covar)).transpose()
                               - expected_return).flatten()

        def mean_variance_optimizer():
            optimization_res = sc_opt.minimize(mean_variance_obj_fun, current_weight, method='SLSQP',
                                               jac=mean_variance_gradient, bounds=general_bnds,
                                               constraints=general_cons, options={"ftol": 10**-12})
            if not optimization_res.success:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Mean variance optimization failed, ' + str(optimization_res.message) + temp
                raise OptimizationError(error_message)
            else:
                return optimization_res.x

        opt_dict = {'risk_parity': log_barrier_risk_parity_optimizer,
                    'min_variance': min_variance_optimizer,
                    'mean_variance': mean_variance_optimizer,
                    'all': [log_barrier_risk_parity_optimizer, min_variance_optimizer]}

        if method is not 'all':
            return pd.DataFrame(opt_dict[method](), index=clean_period_prices.columns.values, columns=[method]), \
                   c_m, data_after_processing[1]
        else:
            temp1 = pd.DataFrame(index=clean_period_prices.columns.values, columns=['risk_parity', 'min_variance',
                                                                                    "mean_variance"])
            n = 0
            for f in opt_dict[method]:
                temp1.iloc[:, n] = f()
                n = n + 1
            return temp1, c_m, data_after_processing[1]

