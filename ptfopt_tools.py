# 05/31/2017 By Chuan Xu @ Ricequant V 2.0


import numpy as np
import rqdatac
import scipy.optimize as sc_opt
from math import *
import pandas as pd
# import matplotlib.pyplot as plt


class OptimizationError(Exception):
    def __init__(self, warning_message):
        print(warning_message)


class TestPortfolio:

    def __init__(self, equity_list, equity_type):

        self.el = equity_list
        self.et = equity_type
        self.reset_start_d = None
        self.reset_end_d = None
        self.daily_cum_log_return = None
        self.daily_arithmetic_return = None
        self.annualized_return = None
        self.annualized_vol = None
        self.clean_equity_list = None
        self.kickout_list = None
        self.st_list = None
        self.suspended_list = None
        self.clean_period_prices = None
        self.period_prices = None

    def data_clean(self, equity_list, start_date, end_date):

        if self.et is 'funds':
            period_prices = rqdatac.fund.get_nav(equity_list, start_date, end_date, fields='acc_net_value')
        elif self.et is 'stocks':
            period_prices = rqdatac.get_price(equity_list, start_date, end_date, frequency='1d', fields='close')
        self.period_prices = period_prices
        # Set up the threshhold of elimination
        out_threshold = ceil(period_prices.shape[0] / 2)
        end_date_T = pd.to_datetime(end_date)
        start_date_T = pd.to_datetime(start_date)
        kickout_list = list()
        suspended_list = list()
        # Locate the first valid value of each column, if available sequence length is less than threshhold, add
        # the column name into out_list; if sequence length is longer than threshold but less than chosen period length,
        # reset the start_date to the later date. The latest start_date whose sequence length is greater than threshold
        # will be chose.
        # Check whether any stocks has long suspended trading periods or has been delisted and generate list
        # for such stocks
        for i in equity_list:
            if rqdatac.is_suspended(i, start_date_T, end_date_T) is not None:
                if ((end_date_T - period_prices.loc[:, i].first_valid_index()) / np.timedelta64(1, 'D')) < out_threshold:
                    kickout_list.append(i)
                elif period_prices.loc[:, i].first_valid_index() < start_date_T:
                    start_date_T = period_prices.loc[:, i].first_valid_index()
                elif rqdatac.is_suspended(i, start_date_T, end_date_T).tail(1).index < end_date_T or \
                                int(rqdatac.is_suspended(i, start_date_T, end_date_T).sum(axis=0)) >= out_threshold:
                    suspended_list.append(i)
            else:
                kickout_list.append(i)
        # Check whether any ST stocks are included and generate a list for ST stocks
        st_list = list(period_prices.columns.values[rqdatac.is_st_stock(equity_list, start_date_T, end_date_T).sum(axis=0)>0])
        # Generate final kickout list which includes all the above
        kickout_list_s = set(kickout_list)
        st_list_s = set(st_list)
        suspended_list_s = set(suspended_list)
        two_list_union = st_list_s.union(suspended_list_s)
        final_dif = two_list_union - kickout_list_s
        final_kickout_list = kickout_list + list(final_dif)
        # Generate clean data
        equity_list_s = set(equity_list)
        final_kickout_list_s = set(final_kickout_list)
        clean_equity_list = list(equity_list_s - final_kickout_list_s)
        if self.et is 'funds':
            clean_period_prices = rqdatac.fund.get_nav(clean_equity_list, start_date_T, end_date_T,
                                                       fields='acc_net_value')
        elif self.et is 'stocks':
            clean_period_prices = rqdatac.get_price(clean_equity_list, start_date_T, end_date_T, frequency='1d',
                                                    fields='close')
        self.clean_period_prices = clean_period_prices
        self.clean_equity_list = list(clean_period_prices.columns.values)
        self.kickout_list = kickout_list
        self.st_list = st_list
        self.suspended_list = suspended_list
        self.reset_start_d = start_date_T
        self.reset_end_d = end_date_T

    def log_barrier_risk_parity_optimizer(self):

        period_daily_return_pct_change = self.clean_period_prices.pct_change() * 100
        c_m = period_daily_return_pct_change.cov()
        x0 = [1 / c_m.shape[0]] * c_m.shape[0]
        log_barrier_risk_parity_obj_fun = lambda x: np.dot(np.dot(x, c_m), x) - 15 * sum(np.log(x))
        log_barrier_bnds = []
        for i in range(len(x0)):
            log_barrier_bnds = log_barrier_bnds + [(0.00001, 1)]
        log_barrier_bnds = tuple(log_barrier_bnds)
        log_barrier_risk_parity_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, x0, method='L-BFGS-B',
                                                           bounds=log_barrier_bnds, options={'maxfun': 30000})
        if not log_barrier_risk_parity_res.success:
            temp = ' @ %s' % self.clean_period_prices.index[0]
            error_message = str(log_barrier_risk_parity_res.message) + temp
            raise OptimizationError(error_message)
        else:
            log_barrier_weights = (log_barrier_risk_parity_res.x / sum(log_barrier_risk_parity_res.x))
            return log_barrier_weights

    def min_variance_optimizer(self):

        period_daily_return_pct_change = self.clean_period_prices.pct_change() * 100
        c_m = period_daily_return_pct_change.cov()
        x0 = [1 / c_m.shape[0]] * c_m.shape[0]
        min_variance_obj_fun = lambda x: np.dot(np.dot(x, c_m), x)
        min_variance_cons_fun = lambda x: sum(x) - 1
        min_variance_cons = ({'type': 'eq', 'fun': min_variance_cons_fun})
        min_variance_bnds = []
        for i in range(len(x0)):
            min_variance_bnds = min_variance_bnds + [(0, 1)]
        min_variance_bnds = tuple(min_variance_bnds)
        SLSQP_min_variance_res = sc_opt.minimize(min_variance_obj_fun, x0, method='SLSQP', bounds=min_variance_bnds,
                                                 constraints=min_variance_cons)
        if not SLSQP_min_variance_res.success:
            temp = ' @ %s' % self.clean_period_prices.index[0]
            error_message = str(SLSQP_min_variance_res.message) + temp
            raise OptimizationError(error_message)
        else:
            SLSQP_min_variance_weights = SLSQP_min_variance_res.x
            return SLSQP_min_variance_weights

    def min_variance_risk_parity_optimizer(self, tol=None):
        period_daily_return_pct_change = self.clean_period_prices.pct_change() * 100
        c_m = period_daily_return_pct_change.cov()
        x0 = [1 / c_m.shape[0]] * c_m.shape[0]
        beta = 0.5
        rho = 1000
        if tol is None:
            tol = 10 ** (-4)
        min_variance_risk_parity_obj_fun = lambda x: (sum((x * np.dot(x, c_m) - sum(x * np.dot(c_m, x))
                                                           / c_m.shape[0]) ** 2) + rho * np.dot(np.dot(x, c_m), x))
        min_variance_risk_parity_cons_fun = lambda x: sum(x) - 1
        min_variance_risk_parity_cons = ({'type': 'eq', 'fun': min_variance_risk_parity_cons_fun})
        min_vairance_risk_parity_bnds = []
        for i in range(len(x0)):
            min_vairance_risk_parity_bnds = min_vairance_risk_parity_bnds + [(-1, 2)]
        min_vairance_risk_parity_bnds = tuple(min_vairance_risk_parity_bnds)
        while rho > tol:
            min_variance_risk_parity_res = sc_opt.minimize(min_variance_risk_parity_obj_fun, x0, method='SLSQP',
                                                           bounds=min_vairance_risk_parity_bnds,
                                                           constraints=min_variance_risk_parity_cons,
                                                           options={'maxiter': 10000})
            if not min_variance_risk_parity_res.success:
                temp = ' @ %s' % self.clean_period_prices.index[0]
                error_message = str(min_variance_risk_parity_res.message) + temp
                raise OptimizationError(error_message)
            x0 = min_variance_risk_parity_res.x
            rho = rho * beta
        x0 = min_variance_risk_parity_res.x
        rho = 0
        min_variance_risk_parity_res = sc_opt.minimize(min_variance_risk_parity_obj_fun, x0, method='SLSQP',
                                                       bounds=min_vairance_risk_parity_bnds,
                                                       constraints=min_variance_risk_parity_cons,
                                                       options={'maxiter': 10000}
                                                       )
        if not min_variance_risk_parity_res.success:
            temp = ' @ %s' % self.clean_period_prices.index[0]
            error_message = str(min_variance_risk_parity_res.message) + temp
            raise OptimizationError(error_message)
        else:
            min_variance_risk_parity_weights = min_variance_risk_parity_res.x
            return min_variance_risk_parity_weights

    def perf_update(self, weights, start_date, end_date):

        if self.kickout_list is not None or self.st_list is not None or self.suspended_list is not None:
            elimination_list = self.kickout_list+self.st_list+self.suspended_list
        else:
            elimination_list = None
        if self.clean_equity_list is None:
            sample_list = self.el
        else:
            sample_list = self.clean_equity_list+elimination_list
        if self.et is 'funds':
            period_prices = rqdatac.fund.get_nav(sample_list, start_date, end_date, fields='acc_net_value')
        elif self.et is 'stocks':
            period_prices = rqdatac.get_price(sample_list, start_date, end_date, frequency='1d', fields='close')
        period_daily_return_pct_change = period_prices.pct_change()[1:]
        new_daily_arithmetic_return = period_daily_return_pct_change.multiply(weights).sum(axis=1)
        if self.daily_arithmetic_return is None:
            self.daily_arithmetic_return = new_daily_arithmetic_return
        else:
            self.daily_arithmetic_return = self.daily_arithmetic_return.append(new_daily_arithmetic_return)
        new_daily_cum_log_return = np.log(new_daily_arithmetic_return + 1).cumsum()
        if self.daily_cum_log_return is None:
            self.daily_cum_log_return = new_daily_cum_log_return
        else:
            self.daily_cum_log_return = self.daily_cum_log_return.append(new_daily_cum_log_return+self.daily_cum_log_return[-1])
        temp = np.log(self.daily_arithmetic_return + 1)
        self.annualized_vol = sqrt(244) * temp.std()
        days_count = (self.daily_arithmetic_return.index[-1] - self.daily_arithmetic_return.index[0]).days
        self.annualized_return = (self.daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1

# def portfolio_performance_gen(portfolio, weights, c_m):
#     daily_return_pct_change = portfolio.pct_change()
#     daily_ptf_arithmetic_return = daily_return_pct_change.multiply(weights).sum(axis=1)
#     daily_ptf_log_return = np.log(daily_ptf_arithmetic_return + 1).cumsum()
#     annualized_ptf_return_std = sqrt(244) * daily_ptf_arithmetic_return.std()
#     days_count = (portfolio.index[-1] - portfolio.index[0]).days
#     annualized_ptf_return = (np.log(daily_ptf_arithmetic_return[1:] + 1).sum() + 1) ** (365 / days_count) - 1
#     risk_ctr = np.dot(weights, c_m) * weights
#     return (daily_ptf_log_return, annualized_ptf_return, annualized_ptf_return_std, risk_ctr)


# def portfolio_performance_plot_gen(equity_list, start_date, end_date, ptf_type, weights=None):
#     portfolio = funds_data_input(equity_list, start_date, end_date)
#     x0 = [1 / portfolio.shape[1]] * portfolio.shape[1]
#     ptf_raw_cov = raw_covariance_cal(portfolio)
#     if weights is None:
#         log_barrier_risk_parity_opt_res = log_barrier_risk_parity_optimizer(ptf_raw_cov)
#         min_variance_opt_res = min_variance_optimizer(ptf_raw_cov)
#         min_variance_risk_parity_opt_res = min_variance_risk_parity_optimizer(ptf_raw_cov)
#         log_barrier_risk_parity_opt_ptf_perf = portfolio_performance_gen(portfolio, log_barrier_risk_parity_opt_res[0],
#                                                                          ptf_raw_cov)
#         min_variance_opt_ptf_perf = portfolio_performance_gen(portfolio, min_variance_opt_res[0], ptf_raw_cov)
#         min_variance_risk_parity_opt_ptf_perf = portfolio_performance_gen(portfolio,
#                                                                           min_variance_risk_parity_opt_res[0],
#                                                                           ptf_raw_cov)
#         equal_weights_opt_ptf_perf = portfolio_performance_gen(portfolio, x0, ptf_raw_cov)
#         fig1 = plt.figure(1)
#         str1 = """Equaity weights portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (equal_weights_opt_ptf_perf[1], equal_weights_opt_ptf_perf[2], ','.join(map(str, x0)),
#                 ','.join(map(str, np.round(equal_weights_opt_ptf_perf[3], decimals=4))))
#         str2 = """Min variance portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (min_variance_opt_ptf_perf[1], min_variance_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(min_variance_opt_res[0], decimals=4))),
#                 ','.join(map(str, np.round(min_variance_opt_ptf_perf[3], decimals=4))))
#         str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (log_barrier_risk_parity_opt_ptf_perf[1],
#                 log_barrier_risk_parity_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(log_barrier_risk_parity_opt_res[0], decimals=4))),
#                 ','.join(map(str, np.round(log_barrier_risk_parity_opt_ptf_perf[3], decimals=4))))
#         str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s].""" % \
#                (min_variance_risk_parity_opt_ptf_perf[1],
#                 min_variance_risk_parity_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(min_variance_risk_parity_opt_res[0], decimals=4))),
#                 ','.join(map(str, np.round(min_variance_risk_parity_opt_ptf_perf[3], decimals=4))))
#         plt.figtext(0.12, 0.03, str1 + str2 + str3 + str4)
#         plt.title('%s to %s In sample %s portfolio performance' %
#                   (portfolio.index[0].date(), portfolio.index[-1].date(), ptf_type))
#         ax = fig1.add_subplot(111)
#         plt.subplots_adjust(bottom=0.18)
#         ax.plot(portfolio.index, equal_weights_opt_ptf_perf[0], label='Equal weights')
#         ax.plot(portfolio.index, min_variance_opt_ptf_perf[0], label='Min variance')
#         ax.plot(portfolio.index, log_barrier_risk_parity_opt_ptf_perf[0], label='Log barrier risk parity')
#         ax.plot(portfolio.index, min_variance_risk_parity_opt_ptf_perf[0], label='Min variance risk parity')
#         ax.legend()
#         return (log_barrier_risk_parity_opt_res[0],min_variance_opt_res[0],min_variance_risk_parity_opt_res[0])
#     else:
#         log_barrier_risk_parity_opt_ptf_perf = portfolio_performance_gen(portfolio, weights['log barrier'],
#                                                                          ptf_raw_cov)
#         min_variance_opt_ptf_perf = portfolio_performance_gen(portfolio, weights['min variance'], ptf_raw_cov)
#         min_variance_risk_parity_opt_ptf_perf = portfolio_performance_gen(portfolio,
#                                                                           weights['min variance risk parity'],
#                                                                           ptf_raw_cov)
#         equal_weights_opt_ptf_perf = portfolio_performance_gen(portfolio, x0, ptf_raw_cov)
#         fig1 = plt.figure(1)
#         str1 = """Equaity weights portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (equal_weights_opt_ptf_perf[1], equal_weights_opt_ptf_perf[2], ','.join(map(str, x0)),
#                 ','.join(map(str, np.round(equal_weights_opt_ptf_perf[3], decimals=4))))
#         str2 = """Min variance portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (min_variance_opt_ptf_perf[1], min_variance_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(weights['min variance'], decimals=4))),
#                 ','.join(map(str, np.round(min_variance_opt_ptf_perf[3], decimals=4))))
#         str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
#                (log_barrier_risk_parity_opt_ptf_perf[1],
#                 log_barrier_risk_parity_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(weights['log barrier'], decimals=4))),
#                 ','.join(map(str, np.round(log_barrier_risk_parity_opt_ptf_perf[3], decimals=4))))
#         str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s].""" % \
#                (min_variance_risk_parity_opt_ptf_perf[1],
#                 min_variance_risk_parity_opt_ptf_perf[2],
#                 ','.join(map(str, np.round(weights['min variance risk parity'], decimals=4))),
#                 ','.join(map(str, np.round(min_variance_risk_parity_opt_ptf_perf[3], decimals=4))))
#         plt.figtext(0.12, 0.03, str1 + str2 + str3 + str4)
#         plt.title('%s to %s Out of sample %s portfolio performance' %
#                   (portfolio.index[0].date(), portfolio.index[-1].date(), ptf_type))
#         ax = fig1.add_subplot(111)
#         plt.subplots_adjust(bottom=0.18)
#         ax.plot(portfolio.index, equal_weights_opt_ptf_perf[0], label='Equal weights')
#         ax.plot(portfolio.index, min_variance_opt_ptf_perf[0], label='Min variance')
#         ax.plot(portfolio.index, log_barrier_risk_parity_opt_ptf_perf[0], label='Log barrier risk parity')
#         ax.plot(portfolio.index, min_variance_risk_parity_opt_ptf_perf[0], label='Min variance risk parity')
#         ax.legend()
#         return None
