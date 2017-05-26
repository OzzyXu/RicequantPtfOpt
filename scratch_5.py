import numpy as np
import matplotlib.pyplot as plt
import math
import e_t as et
import rqdatac
import ptfopt_tools as pt
rqdatac.init("ricequant", "8ricequant8")

fund_name = "000172"
first_period_s = '2014-01-01'
first_period_e = '2014-06-30'
second_period_s = '2014-07-01'
second_period_e = '2014-12-31'
third_period_s = '2015-01-01'
third_period_e = '2015-06-30'
fourth_period_s = '2015-07-01'
fourth_period_e = '2015-12-31'
fifth_period_s = '2016-01-01'
fifth_period_e = '2016-06-30'
sixth_period_s = '2016-07-01'
sixth_period_e = '2016-12-31'
seventh_period_s = '2017-01-01'
seventh_period_e = '2017-05-20'

portfolio = rqdatac.fund.get_holdings(fund_name, fifth_period_s).dropna()
equity_list5 = list(portfolio.order_book_id)
print(rqdatac.get_price(equity_list5, fourth_period_s, fourth_period_e, frequency='1d', fields='close').shape)
print(rqdatac.get_price(equity_list5, fourth_period_s, fourth_period_e, frequency='1d', fields='close').dropna(axis=1, how='any').shape)
print(rqdatac.get_price(equity_list5, fourth_period_s, fourth_period_e, frequency='1d', fields='close').dropna(axis=1, how='all').shape)


print(pt.min_variance_optimizer(equity_list5, 'stocks',fourth_period_s, fourth_period_e)[0])
print(pt.min_variance_optimizer(equity_list5, 'stocks',fourth_period_s, fourth_period_e)[1])
print(pt.min_variance_optimizer(equity_list5, 'stocks',fourth_period_s, fourth_period_e)[2])


portfolio = rqdatac.fund.get_holdings(fund_name, second_period_s).dropna()
equity_list1 = list(portfolio.order_book_id)
portfolio = rqdatac.fund.get_holdings(fund_name, third_period_s).dropna()
equity_list2 = list(portfolio.order_book_id)
portfolio = rqdatac.fund.get_holdings(fund_name, fourth_period_s).dropna()
equity_list3 = list(portfolio.order_book_id)
portfolio = rqdatac.fund.get_holdings(fund_name, fifth_period_s).dropna()
equity_list4 = list(portfolio.order_book_id)
portfolio = rqdatac.fund.get_holdings(fund_name, sixth_period_s).dropna()
equity_list5 = list(portfolio.order_book_id)
portfolio = rqdatac.fund.get_holdings(fund_name, seventh_period_s).dropna()
equity_list6 = list(portfolio.order_book_id)

equity_fund_portfolio_min_variance = pt.TestPortfolio(equity_list1, 'stocks')
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list1, 'stocks',first_period_s, first_period_e)[0],
                                               second_period_s,second_period_e)
equity_fund_portfolio_min_variance.el = equity_list2
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list2, 'stocks',second_period_s, second_period_e)[0],
                                               third_period_s,third_period_e)
equity_fund_portfolio_min_variance.el = equity_list3
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list3, 'stocks',third_period_s, third_period_e)[0],
                                               fourth_period_s,fourth_period_e)
equity_fund_portfolio_min_variance.el = equity_list4
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list4, 'stocks',fourth_period_s, fourth_period_e)[0],
                                               fifth_period_s,fifth_period_e)
equity_fund_portfolio_min_variance.el = equity_list5
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list5, 'stocks',fifth_period_s, fifth_period_e)[0],
                                               sixth_period_s,sixth_period_e)
equity_fund_portfolio_min_variance.el = equity_list6
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_list6, 'stocks',sixth_period_s, sixth_period_e)[0],
                                               seventh_period_s,seventh_period_e)
print(equity_fund_portfolio_min_variance.daily_cum_log_return)

equity_fund_portfolio_min_variance_risk_parity = et.fund_portfolio(equity_list1)
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list1, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
equity_fund_portfolio_min_variance_risk_parity.el = equity_list2
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list2, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
equity_fund_portfolio_min_variance_risk_parity.el = equity_list3
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list3, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
equity_fund_portfolio_min_variance_risk_parity.el = equity_list4
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list4, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
equity_fund_portfolio_min_variance_risk_parity.el = equity_list5
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list5, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
equity_fund_portfolio_min_variance_risk_parity.el = equity_list6
equity_fund_portfolio_min_variance_risk_parity.perf_update(et.min_variance_risk_parity_optimizer(equity_list6, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)

equity_fund_portfolio_log_barrier_risk_parity = et.fund_portfolio(equity_list1)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list1, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
equity_fund_portfolio_log_barrier_risk_parity.el = equity_list2
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list2, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
equity_fund_portfolio_log_barrier_risk_parity.el = equity_list3
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list3, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.el = equity_list4
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list4, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.el = equity_list5
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list5, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.el = equity_list6
equity_fund_portfolio_log_barrier_risk_parity.perf_update(et.log_barrier_risk_parity_optimizer(equity_list6, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)

period_navs = rqdatac.fund.get_nav(fund_name, start_date = second_period_s, end_date = seventh_period_e, fields='acc_net_value')
period_daily_return_pct_change = period_navs.pct_change()[1:]
daily_cum_log_return = np.log(period_daily_return_pct_change + 1).cumsum()
temp = np.log(period_daily_return_pct_change+ 1)
annualized_vol = math.sqrt(244) * temp.std()
days_count = (period_navs.index[-1] - period_navs.index[0]).days
annualized_return = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1

fig1 = plt.figure(1)
str1 = """Fund performance: r = %f, $\sigma$ = %f. """ % \
       (annualized_return, annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (equity_fund_portfolio_min_variance.annualized_return, equity_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (equity_fund_portfolio_log_barrier_risk_parity.annualized_return,
        equity_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (equity_fund_portfolio_min_variance_risk_parity.annualized_return,
        equity_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Equity fund portfolio performance' % (equity_fund_portfolio_min_variance.daily_cum_log_return.index[0].date(),
                                                          equity_fund_portfolio_min_variance.daily_cum_log_return.index[-1].date()))
plt.plot(daily_cum_log_return.index,
         daily_cum_log_return, label=str1)
plt.plot(equity_fund_portfolio_min_variance.daily_cum_log_return.index,
         equity_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(equity_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         equity_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(equity_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         equity_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()
plt.show()