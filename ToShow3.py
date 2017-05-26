# 05/24/2017 By Chuan Xu @ Ricequant
import rqdatac
import ptfopt_tools as pt
import matplotlib.pyplot as plt

rqdatac.init('ricequant', '8ricequant8')

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']
balanced_funds_list = ['519156', '762001', '160212', '160211', '519983']
fixed_income_funds_list = ['110027', '050011', '519977', '161115', '151002']
index_funds_list = ['100032', '162213', '000311', '090010', '310318']
principal_guaranteed_funds_list = ['000030', '000072', '200016', '000270', '163823']
QDIIs_list = ['000071', '164705', '110031', '160717', '206006']

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

x0 = [0.2, 0.2, 0.2, 0.2, 0.2]

# Equity fund portfolio performance generation
equity_fund_portfolio_min_variance = pt.fund_portfolio(equity_funds_list)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
equity_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(equity_funds_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
equity_fund_portfolio_min_variance_risk_parity = pt.fund_portfolio(equity_funds_list)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
equity_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(equity_funds_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
equity_fund_portfolio_log_barrier_risk_parity = pt.fund_portfolio(equity_funds_list)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
equity_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(equity_funds_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
equity_fund_portfolio_equal_weights = pt.fund_portfolio(equity_funds_list)
equity_fund_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
equity_fund_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
equity_fund_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
equity_fund_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
equity_fund_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
equity_fund_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig1 = plt.figure(1)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (equity_fund_portfolio_equal_weights.annualized_return, equity_fund_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (equity_fund_portfolio_min_variance.annualized_return, equity_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (equity_fund_portfolio_log_barrier_risk_parity.annualized_return,
        equity_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (equity_fund_portfolio_min_variance_risk_parity.annualized_return,
        equity_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Equity fund portfolio performance' % (equity_fund_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          equity_fund_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(equity_fund_portfolio_equal_weights.daily_cum_log_return.index,
         equity_fund_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(equity_fund_portfolio_min_variance.daily_cum_log_return.index,
         equity_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(equity_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         equity_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(equity_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         equity_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

# Balanced fund
balanced_fund_portfolio_min_variance = pt.fund_portfolio(balanced_funds_list)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
balanced_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(balanced_funds_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
balanced_fund_portfolio_min_variance_risk_parity = pt.fund_portfolio(balanced_funds_list)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
balanced_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(balanced_funds_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
balanced_fund_portfolio_log_barrier_risk_parity = pt.fund_portfolio(balanced_funds_list)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
balanced_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(balanced_funds_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
balanced_fund_portfolio_equal_weights = pt.fund_portfolio(balanced_funds_list)
balanced_fund_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
balanced_fund_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
balanced_fund_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
balanced_fund_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
balanced_fund_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
balanced_fund_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig2 = plt.figure(2)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (balanced_fund_portfolio_equal_weights.annualized_return, balanced_fund_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (balanced_fund_portfolio_min_variance.annualized_return, balanced_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (balanced_fund_portfolio_log_barrier_risk_parity.annualized_return,
        balanced_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (balanced_fund_portfolio_min_variance_risk_parity.annualized_return,
        balanced_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Balanced fund portfolio performance' % (balanced_fund_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          balanced_fund_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(balanced_fund_portfolio_equal_weights.daily_cum_log_return.index,
         balanced_fund_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(balanced_fund_portfolio_min_variance.daily_cum_log_return.index,
         balanced_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(balanced_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         balanced_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(balanced_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         balanced_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

# Fixed income fund
fixed_income_fund_portfolio_min_variance = pt.fund_portfolio(fixed_income_funds_list)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
fixed_income_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(fixed_income_funds_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity = pt.fund_portfolio(fixed_income_funds_list)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
fixed_income_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(fixed_income_funds_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity = pt.fund_portfolio(fixed_income_funds_list)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
fixed_income_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(fixed_income_funds_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
fixed_income_fund_portfolio_equal_weights = pt.fund_portfolio(fixed_income_funds_list)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
fixed_income_fund_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig3 = plt.figure(3)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (fixed_income_fund_portfolio_equal_weights.annualized_return, fixed_income_fund_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (fixed_income_fund_portfolio_min_variance.annualized_return, fixed_income_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (fixed_income_fund_portfolio_log_barrier_risk_parity.annualized_return,
        fixed_income_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (fixed_income_fund_portfolio_min_variance_risk_parity.annualized_return,
        fixed_income_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Fixed income fund portfolio performance' % (fixed_income_fund_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          fixed_income_fund_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(fixed_income_fund_portfolio_equal_weights.daily_cum_log_return.index,
         fixed_income_fund_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(fixed_income_fund_portfolio_min_variance.daily_cum_log_return.index,
         fixed_income_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(fixed_income_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         fixed_income_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(fixed_income_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         fixed_income_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

# Index fund
index_fund_portfolio_min_variance = pt.fund_portfolio(index_funds_list)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
index_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(index_funds_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
index_fund_portfolio_min_variance_risk_parity = pt.fund_portfolio(index_funds_list)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
index_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(index_funds_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
index_fund_portfolio_log_barrier_risk_parity = pt.fund_portfolio(index_funds_list)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
index_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(index_funds_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
index_fund_portfolio_equal_weights = pt.fund_portfolio(index_funds_list)
index_fund_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
index_fund_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
index_fund_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
index_fund_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
index_fund_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
index_fund_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig4 = plt.figure(4)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (index_fund_portfolio_equal_weights.annualized_return, index_fund_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (index_fund_portfolio_min_variance.annualized_return, index_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (index_fund_portfolio_log_barrier_risk_parity.annualized_return,
        index_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (index_fund_portfolio_min_variance_risk_parity.annualized_return,
        index_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Index fund portfolio performance' % (index_fund_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          index_fund_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(index_fund_portfolio_equal_weights.daily_cum_log_return.index,
         index_fund_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(index_fund_portfolio_min_variance.daily_cum_log_return.index,
         index_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(index_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         index_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(index_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         index_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

# Principal guaranteed fund
principal_guaranteed_fund_portfolio_min_variance = pt.fund_portfolio(principal_guaranteed_funds_list)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
principal_guaranteed_fund_portfolio_min_variance.perf_update(pt.min_variance_optimizer(principal_guaranteed_funds_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity = pt.fund_portfolio(principal_guaranteed_funds_list)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
principal_guaranteed_fund_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(principal_guaranteed_funds_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity = pt.fund_portfolio(principal_guaranteed_funds_list)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
principal_guaranteed_fund_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(principal_guaranteed_funds_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
principal_guaranteed_fund_portfolio_equal_weights = pt.fund_portfolio(principal_guaranteed_funds_list)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
principal_guaranteed_fund_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig5 = plt.figure(5)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (principal_guaranteed_fund_portfolio_equal_weights.annualized_return, principal_guaranteed_fund_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (principal_guaranteed_fund_portfolio_min_variance.annualized_return, principal_guaranteed_fund_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (principal_guaranteed_fund_portfolio_log_barrier_risk_parity.annualized_return,
        principal_guaranteed_fund_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (principal_guaranteed_fund_portfolio_min_variance_risk_parity.annualized_return,
        principal_guaranteed_fund_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s Principal guaranteed fund portfolio performance' % (principal_guaranteed_fund_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          principal_guaranteed_fund_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(principal_guaranteed_fund_portfolio_equal_weights.daily_cum_log_return.index,
         principal_guaranteed_fund_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(principal_guaranteed_fund_portfolio_min_variance.daily_cum_log_return.index,
         principal_guaranteed_fund_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(principal_guaranteed_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         principal_guaranteed_fund_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(principal_guaranteed_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         principal_guaranteed_fund_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

# QDII
QDII_portfolio_min_variance = pt.fund_portfolio(QDIIs_list)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, first_period_s, first_period_e),
                                               second_period_s,second_period_e)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, second_period_s, second_period_e),
                                               third_period_s,third_period_e)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, third_period_s, third_period_e),
                                               fourth_period_s,fourth_period_e)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, fourth_period_s, fourth_period_e),
                                               fifth_period_s,fifth_period_e)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, fifth_period_s, fifth_period_e),
                                               sixth_period_s,sixth_period_e)
QDII_portfolio_min_variance.perf_update(pt.min_variance_optimizer(QDIIs_list, sixth_period_s, sixth_period_e),
                                               seventh_period_s,seventh_period_e)
QDII_portfolio_min_variance_risk_parity = pt.fund_portfolio(QDIIs_list)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, first_period_s, first_period_e),
                                                           second_period_s, second_period_e)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, second_period_s, second_period_e),
                                                           third_period_s, third_period_e)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, third_period_s, third_period_e),
                                                           fourth_period_s, fourth_period_e)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, fourth_period_s, fourth_period_e),
                                                           fifth_period_s, fifth_period_e)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, fifth_period_s, fifth_period_e),
                                                           sixth_period_s, sixth_period_e)
QDII_portfolio_min_variance_risk_parity.perf_update(pt.min_variance_risk_parity_optimizer(QDIIs_list, sixth_period_s, sixth_period_e),
                                                           seventh_period_s, seventh_period_e)
QDII_portfolio_log_barrier_risk_parity = pt.fund_portfolio(QDIIs_list)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, first_period_s, first_period_e),
                                                          second_period_s, second_period_e)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, second_period_s, second_period_e),
                                                          third_period_s, third_period_e)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, third_period_s, third_period_e),
                                                          fourth_period_s, fourth_period_e)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, fourth_period_s, fourth_period_e),
                                                          fifth_period_s, fifth_period_e)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, fifth_period_s, fifth_period_e),
                                                          sixth_period_s, sixth_period_e)
QDII_portfolio_log_barrier_risk_parity.perf_update(pt.log_barrier_risk_parity_optimizer(QDIIs_list, sixth_period_s, sixth_period_e),
                                                          seventh_period_s, seventh_period_e)
QDII_portfolio_equal_weights = pt.fund_portfolio(QDIIs_list)
QDII_portfolio_equal_weights.perf_update(x0,second_period_s,second_period_e)
QDII_portfolio_equal_weights.perf_update(x0,third_period_s,third_period_e)
QDII_portfolio_equal_weights.perf_update(x0,fourth_period_s,fourth_period_e)
QDII_portfolio_equal_weights.perf_update(x0,fifth_period_s,fifth_period_e)
QDII_portfolio_equal_weights.perf_update(x0,sixth_period_s,sixth_period_e)
QDII_portfolio_equal_weights.perf_update(x0,seventh_period_s,seventh_period_e)

fig6 = plt.figure(6)
str1 = """Equal weights portfolio: r = %f, $\sigma$ = %f. """ % \
       (QDII_portfolio_equal_weights.annualized_return, QDII_portfolio_equal_weights.annualized_vol)
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f.""" % \
       (QDII_portfolio_min_variance.annualized_return, QDII_portfolio_min_variance.annualized_vol)
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
       (QDII_portfolio_log_barrier_risk_parity.annualized_return,
        QDII_portfolio_log_barrier_risk_parity.annualized_vol)
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
       (QDII_portfolio_min_variance_risk_parity.annualized_return,
        QDII_portfolio_min_variance_risk_parity.annualized_vol)
plt.title('%s to %s QDII portfolio performance' % (QDII_portfolio_equal_weights.daily_cum_log_return.index[0].date(),
                                                          QDII_portfolio_equal_weights.daily_cum_log_return.index[-1].date()))
plt.plot(QDII_portfolio_equal_weights.daily_cum_log_return.index,
         QDII_portfolio_equal_weights.daily_cum_log_return, label=str1)
plt.plot(QDII_portfolio_min_variance.daily_cum_log_return.index,
         QDII_portfolio_min_variance.daily_cum_log_return, label=str2)
plt.plot(QDII_portfolio_log_barrier_risk_parity.daily_cum_log_return.index,
         QDII_portfolio_log_barrier_risk_parity.daily_cum_log_return, label=str3)
plt.plot(QDII_portfolio_min_variance_risk_parity.daily_cum_log_return.index,
         QDII_portfolio_min_variance_risk_parity.daily_cum_log_return, label=str4)
plt.legend()

fig7 = plt.figure(7)
x1 = [equity_fund_portfolio_min_variance.annualized_vol, index_fund_portfolio_min_variance.annualized_vol,
      principal_guaranteed_fund_portfolio_min_variance.annualized_vol, QDII_portfolio_min_variance.annualized_vol,
      fixed_income_fund_portfolio_min_variance.annualized_vol, balanced_fund_portfolio_min_variance.annualized_vol]
y1 = [equity_fund_portfolio_min_variance.annualized_return, index_fund_portfolio_min_variance.annualized_return,
      principal_guaranteed_fund_portfolio_min_variance.annualized_return, QDII_portfolio_min_variance.annualized_return,
      fixed_income_fund_portfolio_min_variance.annualized_return, balanced_fund_portfolio_min_variance.annualized_return]
x2 = [equity_fund_portfolio_min_variance_risk_parity.annualized_vol, index_fund_portfolio_min_variance_risk_parity.annualized_vol,
      principal_guaranteed_fund_portfolio_min_variance_risk_parity.annualized_vol, QDII_portfolio_min_variance_risk_parity.annualized_vol,
      fixed_income_fund_portfolio_min_variance_risk_parity.annualized_vol, balanced_fund_portfolio_min_variance_risk_parity.annualized_vol]
y2 = [equity_fund_portfolio_min_variance_risk_parity.annualized_return, index_fund_portfolio_min_variance_risk_parity.annualized_return,
      principal_guaranteed_fund_portfolio_min_variance_risk_parity.annualized_return, QDII_portfolio_min_variance_risk_parity.annualized_return,
      fixed_income_fund_portfolio_min_variance_risk_parity.annualized_return, balanced_fund_portfolio_min_variance_risk_parity.annualized_return]
x3 = [equity_fund_portfolio_log_barrier_risk_parity.annualized_vol, index_fund_portfolio_log_barrier_risk_parity.annualized_vol,
      principal_guaranteed_fund_portfolio_log_barrier_risk_parity.annualized_vol, QDII_portfolio_log_barrier_risk_parity.annualized_vol,
      fixed_income_fund_portfolio_log_barrier_risk_parity.annualized_vol, balanced_fund_portfolio_log_barrier_risk_parity.annualized_vol]
y3 = [equity_fund_portfolio_log_barrier_risk_parity.annualized_return, index_fund_portfolio_log_barrier_risk_parity.annualized_return,
      principal_guaranteed_fund_portfolio_log_barrier_risk_parity.annualized_return, QDII_portfolio_log_barrier_risk_parity.annualized_return,
      fixed_income_fund_portfolio_log_barrier_risk_parity.annualized_return, balanced_fund_portfolio_log_barrier_risk_parity.annualized_return]
x4 = [equity_fund_portfolio_equal_weights.annualized_vol, index_fund_portfolio_equal_weights.annualized_vol,
      principal_guaranteed_fund_portfolio_equal_weights.annualized_vol, QDII_portfolio_equal_weights.annualized_vol,
      fixed_income_fund_portfolio_equal_weights.annualized_vol, balanced_fund_portfolio_equal_weights.annualized_vol]
y4 = [equity_fund_portfolio_equal_weights.annualized_return, index_fund_portfolio_equal_weights.annualized_return,
      principal_guaranteed_fund_portfolio_equal_weights.annualized_return, QDII_portfolio_equal_weights.annualized_return,
      fixed_income_fund_portfolio_equal_weights.annualized_return, balanced_fund_portfolio_equal_weights.annualized_return]
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
a = plt.scatter(x1,y1)
b = plt.scatter(x2,y2,c='r',marker='^')
c = plt.scatter(x3,y3,c='g',marker='*')
d = plt.scatter(x4,y4,c='y',marker='s')
plt.legend((a,b,c,d),('Min variance', 'Min variance risk parity', 'Log barrier risk parity', 'Equal weights'))
plt.show()