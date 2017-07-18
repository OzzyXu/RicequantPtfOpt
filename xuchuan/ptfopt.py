# 07/14/2017 By Chuan Xu @ Ricequant V 5.0
import numpy as np
import rqdatac
import scipy.optimize as sc_opt
from math import *
import pandas as pd
import scipy.spatial as scsp

rqdatac.init('ricequant', '8ricequant8')


class OptimizationError(Exception):

    def __init__(self, warning_message):
        print(warning_message)


def data_process(order_book_ids, asset_type, start_date, windows, data_freq, out_threshold_coefficient=None):
    """
    Clean data for covariance matrix calculation
    :param order_book_ids: str list. A group of assets.
    :param asset_type: str. "fund" or "stock"
    :param start_date: str. The first day for backtest.
    :param windows: int. Interval length for sample.
    :param out_threshold_coefficient: float, optional. Determine the threshold to filter out assets with too short data
    which may cause problem in covariance matrix calculation. Whose data length is shorter than threshold will
    be eliminated. Default: 0.5(out_threshold = 0.5*windows).
    :param data_freq: str. Support input: "D": daily data; "W": weekly data; "M": monthly data.
    Weekly data means the close price at the end of each week is taken; monthly means the close price at the end of each
    month. When weekly and monthly data are used, suspended days issues will not be considered. In addition, weekly and
    monthly data don't consider public holidays which have no trading. Users should use a windows a little bit larger
    to get desired data length.
    Users should be very careful when using weekly or monthly data to avoid the observations have too short length.
    :return:
    pandas DataFrame. Contain the prices after cleaning;
    pandas DataFrame. The order_book_ids filtered out and the reasons of elimination;
    str. A new start date for covariance calculation which may differ from default windows setting.
    """

    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    # Choose the start date based on the windows inputted, can't work if backtest start date is earlier than
    # "1995-01-01". The windows for weekly and monthly data don't consider any public holidays which have no trading.
    windows_dict = {"D": -(windows + 1),
                    "W": -(windows + 1) * 5,
                    "M": -(windows + 1) * 22}
    start_date = rqdatac.get_trading_dates("1995-01-01", end_date)[windows_dict[data_freq]]
    reset_start_date = pd.to_datetime(start_date)

    if asset_type is 'fund':
        period_prices = rqdatac.fund.get_nav(order_book_ids, reset_start_date, end_date, fields='adjusted_net_value')
    elif asset_type is 'stock':
        period_data = rqdatac.get_price(order_book_ids, reset_start_date, end_date, frequency='1d',
                                        fields=['close', 'volume'])

        period_prices = period_data['close']
        period_volume = period_data['volume']

    if data_freq is not "D":
        period_prices = period_prices.asfreq(data_freq, method="pad")

    # Set up the threshold of elimination
    if out_threshold_coefficient is None:
        out_threshold = ceil(windows * 0.5)
    else:
        out_threshold = ceil(windows * out_threshold_coefficient)

    kickout_assets = pd.DataFrame(columns=["剔除原因"])

    # Check whether any stocks has long suspended trading periods, have been delisted or new-listed for less than 132
    # trading days and generate list for such stocks. For weekly and monthly data, only those assets which have too late
    # beginning date, were delisted or new-listed will be eliminated.
    if asset_type is "stock":
        if data_freq is "D":
            for i in order_book_ids:
                period_volume_i = period_volume.loc[:, i]
                period_volume_i_value_counts = period_volume_i.value_counts()
                period_volume_i_value_counts_index = period_volume_i_value_counts.index.values
                instrument_i_de_listed_date = rqdatac.instruments(i).de_listed_date
                instrument_i_listed_date = pd.to_datetime(rqdatac.instruments(i).listed_date)
                if not period_volume_i_value_counts.empty:
                    # New-listed stock test
                    if (end_date - instrument_i_listed_date).days <= 132:
                        temp = pd.DataFrame({"剔除原因": "上市时间少于132个交易日"}, index=[i])
                        kickout_assets = kickout_assets.append(temp)
                    # Delisted test
                    elif instrument_i_de_listed_date != "0000-00-00":
                        if pd.to_datetime(instrument_i_de_listed_date) < end_date:
                            temp = pd.DataFrame({"剔除原因": "已退市"}, index=[i])
                            kickout_assets = kickout_assets.append(temp)
                    # Long suspended test
                    elif 0 in period_volume_i_value_counts_index:
                        if period_volume_i_value_counts[period_volume_i_value_counts_index == 0][0] >= out_threshold:
                            temp = pd.DataFrame({"剔除原因": "停牌交易日数量过多"}, index=[i])
                            kickout_assets = kickout_assets.append(temp)
                    # Late beginning day test and just-in-case test for missing values
                    elif period_volume_i.isnull().sum() >= out_threshold:
                        temp = pd.DataFrame({"剔除原因": "缺失值过多"}, index=[i])
                        kickout_assets = kickout_assets.append(temp)
                else:
                    temp = pd.DataFrame({"剔除原因": "无相关股票数据"}, index=[i])
                    kickout_assets = kickout_assets.append(temp)
        else:
            for i in order_book_ids:
                period_prices_i = period_prices.loc[:, i]
                instrument_i_de_listed_date = rqdatac.instruments(i).de_listed_date
                instrument_i_listed_date = pd.to_datetime(rqdatac.instruments(i).listed_date)
                if not ((period_prices_i.isnull() == 0).sum() == 0):
                    # New-listed test
                    if (end_date - instrument_i_listed_date).days <= 132:
                        temp = pd.DataFrame({"剔除原因": "发行时间少于132个交易日"}, index=[i])
                        kickout_assets = kickout_assets.append(temp)
                    # Delisted test
                    elif instrument_i_de_listed_date != "0000-00-00":
                        if pd.to_datetime(instrument_i_de_listed_date) < end_date:
                            temp = pd.DataFrame({"剔除原因": "已退市"}, index=[i])
                            kickout_assets = kickout_assets.append(temp)
                    # Late beginning day test and just-in-case test for missing values
                    elif period_prices_i.isnull().sum() >= out_threshold:
                        temp = pd.DataFrame({"剔除原因": "缺失值过多"}, index=[i])
                        kickout_assets = kickout_assets.append(temp)
                else:
                    temp = pd.DataFrame({"剔除原因": "无相关股票数据"}, index=[i])
                    kickout_assets = kickout_assets.append(temp)

        # # Check whether any ST stocks are included and generate a list for ST stocks
        # st_list = list(period_prices.columns.values[rqdatac.is_st_stock(order_book_ids,
        #                                                                 reset_start_date, end_date).sum(axis=0) > 0])
        # kickout_assets = kickout_assets.append(pd.DataFrame(["ST stocks"] * len(st_list),
        #                                                     columns=["剔除原因"], index=[st_list]))
    elif asset_type is "fund":
        for i in order_book_ids:
            period_prices_i = period_prices.loc[:, i]
            if not ((period_prices_i.isnull() == 0).sum() == 0):
                if period_prices_i.isnull().sum() >= out_threshold:
                    temp = pd.DataFrame({"剔除原因": "缺失值过多"}, index=[i])
                    kickout_assets = kickout_assets.append(temp)
            else:
                temp = pd.DataFrame({"剔除原因": "无相关基金数据"}, index=[i])
                kickout_assets = kickout_assets.append(temp)

    period_prices = period_prices.fillna(method="pad")
    # Generate final kickout list which includes all the above
    final_kickout_list = list(set(kickout_assets.index))
    # Generate clean data and keep the original input id order
    clean_order_book_ids = list(set(order_book_ids) - set(final_kickout_list))

    clean_period_prices = period_prices.loc[reset_start_date:end_date, clean_order_book_ids]
    return clean_period_prices, kickout_assets, reset_start_date


def cov_shrinkage(clean_period_prices):
    """
    Based on Ledoit and Wolf(2003). No r.v.s should have zero sample variance.
    :param clean_period_prices: pandas DataFrame. Sample data after clean.
    :return: pandas DataFrame. Covariance matrix after shrinkage.
             float. Optimal shrinkage intensity.
    """

    cov_m = clean_period_prices.pct_change().cov()
    cov_size = cov_m.shape[0]

    # Generate desired shrinkage target matrix F
    diag_std_m = np.multiply(np.eye(cov_size), np.power(np.diag(cov_m), -0.5))
    corr_m = np.dot(diag_std_m, np.dot(cov_m, diag_std_m))
    corr_avg = 2 * (np.triu(corr_m).sum() - cov_size) / ((cov_size - 1) * cov_size)
    diag_std_v = np.power(np.diag(cov_m), 0.5)
    diag_std_v = diag_std_v[:, None]
    F = np.dot(diag_std_v, diag_std_v.T)
    F_real = np.multiply(np.ones((cov_size, cov_size)) - np.eye(cov_size), corr_avg * F) + np.multiply(np.diag(F),
                                                                                                       np.eye(cov_size))

    # Generate estimator gamma
    gamma_estimator = np.subtract(F_real, cov_m).pow(2).sum().sum()

    # Generate estimator pi
    sample_average_v = clean_period_prices.pct_change().mean()
    pct_after_subtract_m = clean_period_prices.pct_change().subtract(sample_average_v)
    pi_estimator = 0
    pi_ii_list = list()
    v_estimator = np.empty((0, cov_size), float)
    for i in range(cov_size):
        # Calculate estimator pi
        pct_after_subtract_m_i = pct_after_subtract_m.iloc[:, i].values
        temp = np.multiply(pct_after_subtract_m_i, pct_after_subtract_m.T).T
        temp1 = np.subtract(temp, np.array(cov_m.iloc[i, :]))
        temp2 = temp1.pow(2).mean()
        pi_ii_list.append(temp2[i])
        temp3 = temp2.sum()
        pi_estimator += temp3
        # Calculate estimator v for pho
        temp4 = np.subtract(np.power(pct_after_subtract_m_i, 2), cov_m.iloc[i, i])
        temp5 = np.multiply(temp4, temp1.T).T
        v_estimator_i = np.array([temp5.mean()])
        v_estimator = np.append(v_estimator, v_estimator_i, axis=0)

    # Generate estimator pho
    temp = 0
    for i in range(cov_size):
        temp1 = np.multiply(np.delete(diag_std_v, i), np.delete(v_estimator[i, :], i)) / diag_std_v[i]
        temp2 = np.divide(np.delete(v_estimator[:, i], i), np.delete(diag_std_v, i)) * diag_std_v[i]
        temp += sum(temp1+temp2)
    pho_estimator = sum(pi_ii_list) + corr_avg / 2 * temp

    # Generate estimator kai, optimal shrinkage intensity delta and shrinkage target matrix Sigma
    if gamma_estimator != 0:
        kai_estimator = (pi_estimator - pho_estimator) / gamma_estimator
        delta = max(0, min(kai_estimator / clean_period_prices.shape[0], 1))
        return delta * F_real + (1 - delta) * cov_m, delta
    else:
        return cov_m, 0


def black_litterman_prep(order_book_ids, start_date, investors_views, investors_views_indicate_M,
                         investors_views_uncertainty=None, asset_type=None, market_weight=None,
                         risk_free_rate_tenor=None, risk_aversion_coefficient=None, excess_return_cov_uncertainty=None,
                         confidence_of_views=None, windows=None, data_freq=None):
    """
    Generate expected return and expected return covariance matrix with Black-Litterman model. Suppose we have N assets
    and K views. The method can only support daily data so far.
    It's highly recommended to use your own ways to create investors_views_uncertainty, risk_aversion_coefficient and
    excess_return_cov_uncertainty beforehand to get the desired distribution parameters.
    :param order_book_ids: str list. A group of assets;
    :param asset_type: str. "fund" or "stock";
    :param start_date: str. The first day of backtest period;
    :param windows: int. Interval length of sample; Default: 132;
    :param investors_views: K*1 numpy matrix. Each row represents one view;
    :param investors_views_indicate_M: K*N numpy matrix. Each row corresponds to one view. Indicate which view is
    involved during calculation;
    :param investors_views_uncertainty: K*K diagonal matrix, optional. If it is skipped, He and Litterman's method will
    be called to generate diagonal matrix if confidence_of_view is also skipped; Idzorek's method will be called if
    confidence_of_view is passed in; Has to be non-singular;
    :param market_weight: floats list, optional. Weights for market portfolio; Default: Equal weights portfolio;
    :param risk_free_rate_tenor: str, optional. Risk free rate term. Default: "0S"; Support input: "0S", "1M", "3M",
    "6M", "1Y";
    :param risk_aversion_coefficient: float, optional. If no risk_aversion_coefficient is passed in, then
    risk_aversion_coefficient = market portfolio risk premium / market portfolio volatility;
    :param excess_return_cov_uncertainty: float, optional. Default: 1/T where T is the time length of sample;
    :param confidence_of_views: floats list, optional. Represent investors' confidence levels on each view.
    :param data_freq: str. Support input: "D": daily data; "W": weekly data; "M": monthly data.
    Weekly data means the close price at the end of each week is taken; monthly means the close price at the end of each
    month. When weekly and monthly data are used, suspended days issues will not be considered. In addition, weekly and
    monthly data don't consider public holidays which have no trading. Users should use a windows a little bit larger
    to get desired data length.
    Users should be very careful when using weekly or monthly data to avoid the observations have too short length.
    :return:
    numpy matrix. Expected return vector;
    numpy matrix. Covariance matrix of expected return;
    float. risk_aversion_coefficient;
    numpy ndarray. investors_views_uncertainty.
    """

    risk_free_rate_dict = {'0S': 1,
                           '1M': 30,
                           '3M': 92,
                           '6M': 183,
                           '1Y': 365}

    if market_weight is None:
        market_weight = pd.DataFrame([1 / len(order_book_ids)] * len(order_book_ids), index=order_book_ids)
    if windows is None:
        windows = 132
    if data_freq is None:
        data_freq = "D"
    if asset_type is None:
        asset_type = "fund"
    if risk_free_rate_tenor is None:
        risk_free_rate_tenor = "0S"

    # Clean data
    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    clean_period_prices, reset_start_date = (data_process(order_book_ids, asset_type, start_date, windows, data_freq)[i]
                                             for i in [0, 2])

    if excess_return_cov_uncertainty is None:
        excess_return_cov_uncertainty = 1 / clean_period_prices.shape[0]

    # Fetch risk free rate data
    reset_start_date = rqdatac.get_next_trading_date(reset_start_date)
    risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor=risk_free_rate_tenor, country='cn')
    if data_freq is not "D":
        risk_free_rate = risk_free_rate.asfreq(data_freq, method="pad")
    risk_free_rate[data_freq] = pd.Series(np.power(1 + risk_free_rate.iloc[:, 0],
                                                   risk_free_rate_dict[risk_free_rate_tenor] / 365) - 1,
                                          index=risk_free_rate.index)

    # Calculate risk premium for each equity
    clean_period_prices_pct_change = clean_period_prices.pct_change()
    clean_period_excess_return = clean_period_prices_pct_change.subtract(risk_free_rate[data_freq], axis=0)

    # Wash out the ones in kick_out_list
    clean_market_weight = market_weight.loc[clean_period_prices.columns.values]
    temp_sum_weight = clean_market_weight.sum()
    clean_market_weight = clean_market_weight.div(temp_sum_weight)

    # If no risk_aversion_coefficient is passed in, then
    # risk_aversion_coefficient = market portfolio risk premium / market portfolio volatility
    if risk_aversion_coefficient is None:
        market_portfolio_return = np.dot(clean_period_prices_pct_change, clean_market_weight)
        risk_aversion_coefficient = ((market_portfolio_return[1:].mean() - risk_free_rate[data_freq].mean()) /
                                     market_portfolio_return[1:].var())

    clean_period_excess_return_cov = clean_period_excess_return[1:].cov()
    equilibrium_return = np.multiply(np.dot(clean_period_excess_return_cov, clean_market_weight),
                                     risk_aversion_coefficient)

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
                                                                           investors_views_indicate_M[i, :].T)))
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
                omega_k = sc_opt.minimize_scalar(objective_fun, bounds=(10 **-8, upper_bound), method="bounded",
                                                 options={"xatol": 10**-8})
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


# Generate upper and lower bounds for assets in portfolio
def bounds_gen(order_book_ids, clean_order_book_ids, method, bounds=None):

    if bounds is not None:
        # Bounds setup error check
        temp_lb = 0
        for key in bounds:
            if bounds[key][0] > bounds[key][1]:
                raise OptimizationError("Lower bound is larger than upper bound for asset %s." % key)
            elif bounds[key][0] > 1 or bounds[key][1] < 0:
                raise OptimizationError("Bounds setting error for %s" % key)
            if key is not "full_list":
                if key not in order_book_ids:
                    raise OptimizationError('Bounds setting contains asset %s who doesnt exist in assets pool.' % key)
                if method is not "risk_parity":
                    temp_lb += bounds[key][0]
            else:
                if method is not "risk_parity":
                    temp_lb = bounds[key][0] * len(order_book_ids)
        if temp_lb > 1:
            raise OptimizationError("The summation of lower bounds is larger than 1.")

        general_bnds = list()
        log_rp_bnds = list()
        bounds_list = list(bounds)
        temp_ub = 0
        if method is "risk_parity":
            log_rp_bnds = [(10 ** -6, float('inf'))] * len(clean_order_book_ids)
        elif method is "all":
            log_rp_bnds = [(10 ** -6, float('inf'))] * len(clean_order_book_ids)
            for i in clean_order_book_ids:
                if "full_list" in bounds_list:
                    general_bnds = general_bnds + [(max(0, bounds["full_list"][0]), min(1, bounds["full_list"][1]))]
                    temp_ub += bounds["full_list"][1]
                elif i in bounds_list:
                    general_bnds = general_bnds + [(max(0, bounds[i][0]), min(1, bounds[i][1]))]
                    temp_ub += bounds[i][1]
                else:
                    general_bnds = general_bnds + [(0, 1)]
                    temp_ub += 1
        else:
            for i in clean_order_book_ids:
                if "full_list" in bounds_list:
                    general_bnds = general_bnds + [(max(0, bounds["full_list"][0]), min(1, bounds["full_list"][1]))]
                    temp_ub += bounds["full_list"][1]
                elif i in bounds_list:
                    general_bnds = general_bnds + [(max(0, bounds[i][0]), min(1, bounds[i][1]))]
                    temp_ub += bounds[i][1]
                else:
                    general_bnds = general_bnds + [(0, 1)]
                    temp_ub += 1

        if method is not "risk_parity":
            if temp_ub < 1:
                kickout_list = list(set(order_book_ids) - set(clean_order_book_ids))
                message = ("The summation of upper bounds is less than one after data processing! The following assets "
                           "have been eliminated: %s" % kickout_list)
                raise OptimizationError(message)

        if method is "all":
            return tuple(log_rp_bnds), tuple(general_bnds)
        elif method is "risk_parity":
            return tuple(log_rp_bnds)
        else:
            return tuple(general_bnds)
    else:
        log_rp_bnds = [(10 ** -6, float('inf'))] * len(clean_order_book_ids)
        general_bnds = [(0, 1)] * len(clean_order_book_ids)
        if method is "all":
            return tuple(log_rp_bnds), tuple(general_bnds)
        elif method is "risk_parity":
            return tuple(log_rp_bnds)
        else:
            return tuple(general_bnds)


# Market neutral constraints generation
def market_neutral_constraints_gen(clean_order_book_ids, asset_type, market_neutral_constraints, benchmark):

    pass



# Generate category constraints generation
def category_constraints_gen(clean_order_book_ids, asset_type, constraints=None):

    if constraints is not None:
        df = pd.DataFrame(index=clean_order_book_ids, columns=['type'])

        # Constraints setup error check
        temp_lb = 0
        temp_ub = 0
        for key in constraints:
            temp_lb += constraints[key][0]
            temp_ub += constraints[key][1]
            if constraints[key][0] > constraints[key][1]:
                raise OptimizationError("Constraints setup error for %s." % key)
            if constraints[key][0] > 1 or constraints[key][1] < 0:
                raise OptimizationError("Constraints setup error for %s." % key)
        if temp_ub < 1 or temp_lb > 1:
            raise OptimizationError("Constraints summation error.")

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


def optimizer(order_book_ids, start_date, asset_type, method, current_weight=None, bnds=None, cons=None,
              expected_return=None, expected_return_covar=None, risk_aversion_coefficient=1, windows=None,
              out_threshold_coefficient=None, data_freq=None, fun_tol=10**-8, max_iteration=10**3, disp=False,
              iprint=1, cov_enhancement=True, benchmark=None):
    """
    :param order_book_ids: str list. A list of assets(stocks or funds). Optional when expected_return_covar is given;
    :param start_date: str. Date to initialize a portfolio or re-balance a portfolio. Optional when
    expected_return_covar is given;
    :param asset_type: str. "stock" or "fund". Types of portfolio candidates, portfolio with mixed assets is not
    supported;
    :param method: str. Portfolio optimization model: "risk_parity", "min_variance", "mean_variance",
    "risk_parity_with_con", "min_TE", "all"("all" method only contains "risk_parity", "min_variance",
    "risk_parity_with_con"). When "min_TE" method is chosen, expected_return_covar must be None type.
    :param current_weight: floats list, optional. Default: 1/N(N: no. of assets). Initial guess for optimization.
    :param bnds: floats list, optional. Lower bounds and upper bounds for each asset in portfolio.
    Support input format: {"asset_code1": (lb1, up1), "asset_code2": (lb2, up2), ...} or {'full_list': (lb, up)} (set up
    universal bounds for all assets);
    :param cons: dict, optional. Lower bounds and upper bounds for each category of assets in portfolio;
    Supported funds type: Bond, Stock, Hybrid, Money, ShortBond, StockIndex, BondIndex, Related, QDII, Other; supported
    stocks industry sector: Shenwan_industry_name;
    cons: {"types1": (lb1, up1), "types2": (lb2, up2), ...};
    :param expected_return: pandas DataFrame. Default: Means of the returns for order_book_ids
    within windows(empirical means). Must input this if expected_return_covar is given to run "mean_variance" method.
    :param expected_return_covar: pandas DataFrame, optional. Covariance matrix of expected return. Default: covariance
    of the means of the returns of order_book_ids within windows. If expected_return_covar is given, any models involve
    covariance matrix will use expected_return_covar instead of estimating from sample data. Moreover, if
    expected_return_covar is given and "mean_variance" method is chosen, expected_return must also be given;
    :param risk_aversion_coefficient: float, optional. Risk aversion coefficient of Mean-Variance model. Default: 1.
    :param windows: int, optional. Default: 132. Data windows length.
    :param data_freq: str, optional. Default: "D". Support input: "D": daily data; "W": weekly data; "M": monthly data.
    Weekly data means the close price at the end of each week is taken; monthly means the close price at the end of each
    month. When weekly and monthly data are used, suspended days issues will not be considered. In addition, weekly and
    monthly data don't consider public holidays which have no trading. Users should use a windows a little bit larger
    to get desired data length.
    :param out_threshold_coefficient: float, optional. Determine the threshold to filter out assets with too short data
    which may cause problem in covariance matrix calculation. Whose data length is shorter than threshold will
    be eliminated. Default: 0.5(out_threshold = 0.5*windows).
    :param fun_tol: float, optional. Optimization accuracy requirement. The smaller, the more accurate, but cost more
    time. Default: 10E-12.
    :param max_iteration: int, optional. Max iteration number allows during optimization. Default: 1000.
    :param disp: bool, optional. Optimization summary display control. Override iprint interface. Default: False.
    :param cov_enhancement: bool, optional. Default: True. Use shrinkage method based on Ledoit and Wolf(2003) to
    improve the estimation for sample covariance matrix. It's recommended to set it to True when the stock pool is
    large.
    :param benchmark: str, optional. Target to track in minimum tracking error("min_TE") method.
    :param iprint: int, optional.
    The verbosity of optimization:
        * iprint <= 0 : Silent operation;
        * iprint == 1 : Print summary upon completion (default);
        * iprint >= 2 : Print status of each iterate and summary.
    :return:
    pandas DataFrame. A DataFrame contains assets' name and their corresponding optimal weights;
    pandas DataFrame. The covariance matrix for optimization;
    pandas DataFrame. The order_book_ids filtered out and the reasons of elimination;
    str. Optimization message. Return this only when methods other than "all".
    """

    if not disp:
        iprint = 0

    opts = {'maxiter': max_iteration,
            'ftol': fun_tol,
            'iprint': iprint,
            'disp': disp}

    if data_freq is None:
        data_freq = "D"
    if windows is None:
        windows = 132

    if expected_return_covar is None:
        # Get clean data and calculate covariance matrix if no expected_return_covar is given
        data_after_processing = data_process(order_book_ids, asset_type, start_date, windows, data_freq,
                                             out_threshold_coefficient)
        clean_period_prices = data_after_processing[0]
        reset_start_date = data_after_processing[2]

        # If all assets are eliminated, raise error
        if clean_period_prices.shape[1] == 0:
            # print('All selected funds have been ruled out')
            raise OptimizationError("All assets have been eliminated")

        # Generate enhanced estimation for covariance matrix
        period_daily_return_pct_change = clean_period_prices.pct_change()[1:]
        if cov_enhancement:
            c_m = cov_shrinkage(clean_period_prices)[0]
        else:
            c_m = period_daily_return_pct_change.cov()

        # Generate initial guess point with equal weights
        if current_weight is None:
            current_weight = [1 / clean_period_prices.shape[1]] * clean_period_prices.shape[1]
        else:
            new_current_weight = current_weight
            current_weight = list()
            for i in clean_period_prices.columns.values:
                current_weight.append(new_current_weight[order_book_ids.index(i)])

        # Generate expected_return if not given
        if method is "mean_variance":
            empirical_mean = period_daily_return_pct_change.mean()
            if expected_return is None:
                expected_return = empirical_mean
            else:
                for i in expected_return.index.values:
                    if i in empirical_mean.index.values:
                        empirical_mean.loc[i] = expected_return.loc[i]
                expected_return = empirical_mean
    else:
        # Get preparation done when expected_return_covar is given
        c_m = expected_return_covar

        if current_weight is None:
            current_weight = [1 / c_m.shape[0]] * c_m.shape[0]

        order_book_ids = list(c_m.columns.values)

    # Read benchmark data for min tracking error model
    if method is "min_TE":
        if benchmark is None:
            raise OptimizationError("Input no benchmark!")
        benchmark_price = rqdatac.get_price(benchmark, start_date=reset_start_date,
                                            end_date=rqdatac.get_previous_trading_date(start_date), fields="close")
        if data_freq is not "D":
            benchmark_price_change = benchmark_price.asfreq(data_freq, method="pad").pct_change()[1:]
        else:
            benchmark_price_change = benchmark_price.pct_change()[1:]

    # Generate bounds
    clean_order_book_ids = list(c_m.columns.values)
    if method is "all":
        log_rp_bnds, general_bnds = bounds_gen(order_book_ids, clean_order_book_ids, method, bnds)
    elif method is "risk_parity":
        log_rp_bnds = bounds_gen(order_book_ids, clean_order_book_ids, method, bnds)
    else:
        general_bnds = bounds_gen(order_book_ids, clean_order_book_ids, method, bnds)

    # Generate constraints
    general_cons = category_constraints_gen(clean_order_book_ids, asset_type, cons)

    # Log barrier risk parity model
    c = 15

    def log_barrier_risk_parity_obj_fun(x):
        return np.dot(x, np.dot(c_m, x)) - c * sum(np.log(x))

    def log_barrier_risk_parity_gradient(x):
        return np.multiply(2, np.dot(c_m, x)) - np.multiply(c, np.reciprocal(x))

    def log_barrier_risk_parity_optimizer():
        optimization_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, current_weight, method='L-BFGS-B',
                                           jac=log_barrier_risk_parity_gradient, bounds=log_rp_bnds)
        optimization_info = optimization_res.message
        if not optimization_res.success:
            if optimization_res.nit >= max_iteration:
                optimal_weights = (optimization_res.x / sum(optimization_res.x))
                return optimal_weights, optimization_info
            else:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Risk parity optimization failed, ' + str(optimization_res.message) + temp
                raise OptimizationError(error_message)
        else:
            optimal_weights = (optimization_res.x / sum(optimization_res.x))
            return optimal_weights, optimization_info

    # Risk parity with constraints model
    def risk_parity_with_con_obj_fun(x):
        temp1 = np.multiply(x, np.dot(c_m, x))
        temp2 = temp1[:, None]
        return np.sum(scsp.distance.pdist(temp2, "euclidean"))

    def risk_parity_with_con_optimizer():
        optimization_res = sc_opt.minimize(risk_parity_with_con_obj_fun, current_weight, method='SLSQP',
                                           bounds=general_bnds, constraints=general_cons, options=opts)
        optimization_info = optimization_res.message
        if not optimization_res.success:
            if optimization_res.nit >= max_iteration:
                return optimization_res.x, optimization_info
            else:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Risk parity with constraints optimization failed, ' + str(optimization_res.message) \
                                + temp
                raise OptimizationError(error_message)
        else:
            return optimization_res.x, optimization_info

    # Min variance model
    min_variance_obj_fun = lambda x: np.dot(np.dot(x, c_m), x)

    def min_variance_gradient(x):
        return np.multiply(2, np.dot(c_m, x))

    def min_variance_optimizer():
        optimization_res = sc_opt.minimize(min_variance_obj_fun, current_weight, method='SLSQP',
                                           jac=min_variance_gradient, bounds=general_bnds, constraints=general_cons,
                                           options=opts)
        optimization_info = optimization_res.message
        if not optimization_res.success:
            if optimization_res.nit >= max_iteration:
                return optimization_res.x, optimization_info
            else:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Min variance optimization failed, ' + str(optimization_res.message) + temp
                raise OptimizationError(error_message)
        else:
            return optimization_res.x, optimization_info

    # Mean variance model
    def mean_variance_obj_fun(x):
        return (np.multiply(risk_aversion_coefficient / 2, np.dot(np.dot(x, c_m), x)) -
                np.dot(x, expected_return))

    def mean_variance_gradient(x):
        return np.asfarray(np.multiply(risk_aversion_coefficient, np.dot(x, c_m)).transpose()
                           - expected_return).flatten()

    def mean_variance_optimizer():
        optimization_res = sc_opt.minimize(mean_variance_obj_fun, current_weight, method='SLSQP',
                                           jac=mean_variance_gradient, bounds=general_bnds,
                                           constraints=general_cons, options=opts)
        optimization_info = optimization_res.message
        if not optimization_res.success:
            if optimization_res.nit >= max_iteration:
                return optimization_res.x, optimization_info
            else:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Mean variance optimization failed, ' + str(optimization_res.message) + temp
                raise OptimizationError(error_message)
        else:
            return optimization_res.x, optimization_info

    # Minimizing tracking error model
    def min_TE_obj_fun(x):
        return np.dot(np.subtract(benchmark_price_change, np.dot(period_daily_return_pct_change, x)).T,
                      np.subtract(benchmark_price_change, np.dot(period_daily_return_pct_change, x)))

    def min_TE_optimizer():
        optimization_res = sc_opt.minimize(min_TE_obj_fun, current_weight, method='SLSQP',
                                           bounds=general_bnds, constraints=general_cons, options=opts)
        optimization_info = optimization_res.message
        if not optimization_res.success:
            if optimization_res.nit >= max_iteration:
                return optimization_res.x, optimization_info
            else:
                temp = ' @ %s' % clean_period_prices.index[0]
                error_message = 'Min TE optimization failed, ' + str(optimization_res.message) + temp
                raise OptimizationError(error_message)
        else:
            return optimization_res.x, optimization_info

    opt_dict = {'risk_parity': log_barrier_risk_parity_optimizer,
                'min_variance': min_variance_optimizer,
                'mean_variance': mean_variance_optimizer,
                'risk_parity_with_con': risk_parity_with_con_optimizer,
                "min_TE": min_TE_optimizer,
                'all': [log_barrier_risk_parity_optimizer, min_variance_optimizer, risk_parity_with_con_optimizer]}

    if method is not 'all':
        if expected_return_covar is None:
            return pd.DataFrame(opt_dict[method]()[0], index=list(c_m.columns.values), columns=[method]), c_m, \
                   data_after_processing[1], opt_dict[method]()[1]
        else:
            pd.DataFrame(opt_dict[method]()[0], index=list(c_m.columns.values), columns=[method]), c_m, \
            opt_dict[method]()[1]
    else:
        temp1 = pd.DataFrame(index=list(c_m.columns.values), columns=['risk_parity', 'min_variance',
                                                                      "risk_parity_with_con"])
        n = 0
        for f in opt_dict[method]:
            temp1.iloc[:, n] = f()[0]
            n = n + 1
        if expected_return_covar is None:
            return temp1, c_m, data_after_processing[1]
        else:
            return temp1, c_m
