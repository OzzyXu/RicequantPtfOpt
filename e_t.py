import numpy as np
import math
import scipy.optimize as sc_opt
import rqdatac
rqdatac.init("ricequant", "8ricequant8")

def log_barrier_risk_parity_optimizer(equity_list, start_date, end_date):
    period_navs = rqdatac.get_price(equity_list, start_date, end_date, frequency='1d', fields='close')
    period_daily_return_pct_change = period_navs.pct_change()*100
    c_m = period_daily_return_pct_change.cov()
    x0 = [1 / c_m.shape[0]] * c_m.shape[0]
    log_barrier_risk_parity_obj_fun = lambda x: np.dot(np.dot(x, c_m), x) - 15 * sum(np.log(x))
    log_barrier_bnds = []
    for i in range(len(x0)):
        log_barrier_bnds = log_barrier_bnds + [(0, 1)]
    log_barrier_bnds = tuple(log_barrier_bnds)
    BFGS_log_barrier_risk_parity_res = sc_opt.minimize(log_barrier_risk_parity_obj_fun, x0, method='L-BFGS-B',
                                                       bounds=log_barrier_bnds)
    BFGS_log_barrier_weights = (BFGS_log_barrier_risk_parity_res.x / sum(BFGS_log_barrier_risk_parity_res.x))
    return BFGS_log_barrier_weights

def min_variance_optimizer(equity_list, start_date, end_date):
    period_navs = rqdatac.get_price(equity_list, start_date, end_date, frequency='1d', fields='close')
    period_daily_return_pct_change = period_navs.pct_change()*100
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
    SLSQP_min_variance_weights = SLSQP_min_variance_res.x
    return SLSQP_min_variance_weights


def min_variance_risk_parity_optimizer(equity_list, start_date, end_date, tol=None):
    pperiod_navs = rqdatac.get_price(equity_list, start_date, end_date, frequency='1d', fields='close')
    period_daily_return_pct_change = period_navs.pct_change()*100
    c_m = period_daily_return_pct_change.cov()
    x0 = [1 / c_m.shape[0]] * c_m.shape[0]
    beta = 0.5
    rho = 1000
    if tol is None:
        tol = 10 ** (-4)
    min_variance_risk_parity_obj_fun = lambda x: sum(
        (x * np.dot(x, c_m) - sum(x * np.dot(c_m, x)) / c_m.shape[0]) ** 2) + \
                                                 rho * np.dot(np.dot(x, c_m), x)
    min_variance_risk_parity_cons_fun = lambda x: sum(x) - 1
    min_variance_risk_parity_cons = ({'type': 'eq', 'fun': min_variance_risk_parity_cons_fun})
    min_vairance_risk_parity_bnds = []
    for i in range(len(x0)):
        min_vairance_risk_parity_bnds = min_vairance_risk_parity_bnds + [(-1, 2)]
    min_vairance_risk_parity_bnds = tuple(min_vairance_risk_parity_bnds)
    while rho > tol:
        min_variance_risk_parity_res = sc_opt.minimize(min_variance_risk_parity_obj_fun, x0, method='SLSQP',
                                                       bounds=min_vairance_risk_parity_bnds,
                                                       constraints=min_variance_risk_parity_cons)
        x0 = min_variance_risk_parity_res.x
        rho = rho * beta
    x0 = min_variance_risk_parity_res.x
    rho = 0
    min_variance_risk_parity_res = sc_opt.minimize(min_variance_risk_parity_obj_fun, x0, method='SLSQP',
                                                   bounds=min_vairance_risk_parity_bnds,
                                                   constraints=min_variance_risk_parity_cons)
    min_variance_risk_parity_weights = min_variance_risk_parity_res.x
    return min_variance_risk_parity_weights

class fund_portfolio:
    def __init__(self, equity_list):
        self.el = equity_list
        self.daily_cum_log_return = None
        self.daily_arithmetic_return = None
        self.annualized_return = None
        self.annualized_vol = None

    def perf_update(self, weights, start_date, end_date):
        period_navs = rqdatac.get_price(equity_list, start_date, end_date, frequency='1d', fields='close')
        period_daily_return_pct_change = period_navs.pct_change()[1:]
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
        self.annualized_vol = math.sqrt(244) * temp.std()
        days_count = (self.daily_arithmetic_return.index[-1] - self.daily_arithmetic_return.index[0]).days
        self.annualized_return = (self.daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1