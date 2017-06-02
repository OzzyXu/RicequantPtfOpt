import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rqdatac

import ptfopt_tools as pt

rqdatac.init("ricequant", "8ricequant8")

fund_name = "233009"
first_period_s = '2014-01-01'
first_period_e = '2014-06-30'
second_period_s = '2014-07-01'
second_period_e = '2014-12-30'
third_period_s = '2015-01-01'
third_period_e = '2015-06-30'
fourth_period_s = '2015-07-01'
fourth_period_e = '2015-12-30'
fifth_period_s = '2016-01-01'
fifth_period_e = '2016-06-30'
sixth_period_s = '2016-07-01'
sixth_period_e = '2016-12-30'
seventh_period_s = '2017-01-01'
seventh_period_e = '2017-05-20'

portfolio1 = rqdatac.fund.get_holdings(fund_name, second_period_s).dropna()
equity_list1 = list(portfolio1.order_book_id)
portfolio2 = rqdatac.fund.get_holdings(fund_name, third_period_s).dropna()
equity_list2 = list(portfolio2.order_book_id)
portfolio3 = rqdatac.fund.get_holdings(fund_name, fourth_period_s).dropna()
equity_list3 = list(portfolio3.order_book_id)
portfolio4 = rqdatac.fund.get_holdings(fund_name, fifth_period_s).dropna()
equity_list4 = list(portfolio4.order_book_id)
portfolio5 = rqdatac.fund.get_holdings(fund_name, sixth_period_s).dropna()
equity_list5 = list(portfolio5.order_book_id)
portfolio6 = rqdatac.fund.get_holdings(fund_name, seventh_period_s).dropna()
equity_list6 = list(portfolio6.order_book_id)

# # Min variance optimizer
# equity_fund_portfolio_min_variance = pt.TestPortfolio(equity_list1, 'stocks')
# equity_fund_portfolio_min_variance.data_clean(equity_list1, first_period_s, first_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio1.loc[portfolio1['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, second_period_s, second_period_e)
#
# equity_fund_portfolio_min_variance.data_clean(equity_list2, second_period_s, second_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio2.loc[portfolio2['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, third_period_s, third_period_e)
#
# equity_fund_portfolio_min_variance.data_clean(equity_list3, third_period_s, third_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio3.loc[portfolio3['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, fourth_period_s, fourth_period_e)
#
# equity_fund_portfolio_min_variance.data_clean(equity_list4, fourth_period_s, fourth_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio4.loc[portfolio4['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, fifth_period_s, fifth_period_e)
#
# equity_fund_portfolio_min_variance.data_clean(equity_list5, fifth_period_s, fifth_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio5.loc[portfolio5['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, sixth_period_s, sixth_period_e)
#
# equity_fund_portfolio_min_variance.data_clean(equity_list6, sixth_period_s, sixth_period_e)
# elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
#                    equity_fund_portfolio_min_variance.suspended_list
# inherited_holdings_weights = list(portfolio6.loc[portfolio6['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance.perf_update(weights, seventh_period_s, seventh_period_e)

# Log barrier risk parity optimizer
equity_fund_portfolio_log_barrier = pt.TestPortfolio(equity_list1, 'stocks')
equity_fund_portfolio_log_barrier.data_clean(equity_list1, first_period_s, first_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio1.loc[portfolio1['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, second_period_s, second_period_e)

equity_fund_portfolio_log_barrier.data_clean(equity_list2, second_period_s, second_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio2.loc[portfolio2['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, third_period_s, third_period_e)

equity_fund_portfolio_log_barrier.data_clean(equity_list3, third_period_s, third_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio3.loc[portfolio3['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, fourth_period_s, fourth_period_e)

equity_fund_portfolio_log_barrier.data_clean(equity_list4, fourth_period_s, fourth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio4.loc[portfolio4['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, fifth_period_s, fifth_period_e)

equity_fund_portfolio_log_barrier.data_clean(equity_list5, fifth_period_s, fifth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio5.loc[portfolio5['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, sixth_period_s, sixth_period_e)

equity_fund_portfolio_log_barrier.data_clean(equity_list6, sixth_period_s, sixth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio6.loc[portfolio6['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
equity_fund_portfolio_log_barrier.perf_update(weights, seventh_period_s, seventh_period_e)


# # Min variance risk parity optimizer
# equity_fund_portfolio_min_variance_risk_parity = pt.TestPortfolio(equity_list1, 'stocks')
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list1, first_period_s, first_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio1.loc[portfolio1['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, second_period_s, second_period_e)
#
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list2, second_period_s, second_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio2.loc[portfolio2['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, third_period_s, third_period_e)
#
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list3, third_period_s, third_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio3.loc[portfolio3['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, fourth_period_s, fourth_period_e)
#
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list4, fourth_period_s, fourth_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio4.loc[portfolio4['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, fifth_period_s, fifth_period_e)
#
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list5, fifth_period_s, fifth_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio5.loc[portfolio5['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, sixth_period_s, sixth_period_e)
#
# equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list6, sixth_period_s, sixth_period_e)
# elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
#                    equity_fund_portfolio_min_variance_risk_parity.suspended_list
# inherited_holdings_weights = list(portfolio6.loc[portfolio6['order_book_id'].isin(elimination_list)].weight)
# inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
# optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
# optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
# weights = optimal_weights+inherited_holdings_weights
# equity_fund_portfolio_min_variance_risk_parity.perf_update(weights, seventh_period_s, seventh_period_e)

# 　Fund net value path
period_navs = rqdatac.fund.get_nav(fund_name, start_date=second_period_s, end_date=seventh_period_e,
                                   fields='acc_net_value')
period_daily_return_pct_change = period_navs.pct_change()[1:]
daily_cum_log_return = np.log(period_daily_return_pct_change + 1).cumsum()
temp = np.log(period_daily_return_pct_change + 1)
annualized_vol = math.sqrt(244) * temp.std()
days_count = (period_navs.index[-1] - period_navs.index[0]).days
annualized_return = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1

# Fund holding path
equity_fund_portfolio_holdings = pt.TestPortfolio(equity_list1, 'stocks')
original_weights = list(portfolio1.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, second_period_s, second_period_e)
equity_fund_portfolio_holdings.el = equity_list2
original_weights = list(portfolio2.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, third_period_s, third_period_e)
equity_fund_portfolio_holdings.el = equity_list3
original_weights = list(portfolio3.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, fourth_period_s, fourth_period_e)
equity_fund_portfolio_holdings.el = equity_list4
original_weights = list(portfolio4.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, fifth_period_s, fifth_period_e)
equity_fund_portfolio_holdings.el = equity_list5
original_weights = list(portfolio5.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, sixth_period_s, sixth_period_e)
equity_fund_portfolio_holdings.el = equity_list6
original_weights = list(portfolio6.weight)
original_weights = [x / 100 for x in original_weights]
equity_fund_portfolio_holdings.perf_update(original_weights, seventh_period_s, seventh_period_e)

fig1 = plt.figure()
str1 = """Fund NAV cumulative return path: r = %f, $\sigma$ = %f. """ % \
       (annualized_return, annualized_vol)
str2 = """Minimum variance portfolio cumulative return path: r = %f, $\sigma$ = %f.""" % \
       (equity_fund_portfolio_min_variance.annualized_return, equity_fund_portfolio_min_variance.annualized_vol)
# str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f. """ % \
#        (equity_fund_portfolio_log_barrier.annualized_return,
#         equity_fund_portfolio_log_barrier.annualized_vol)
# str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f.""" % \
#        (equity_fund_portfolio_min_variance_risk_parity.annualized_return,
#         equity_fund_portfolio_min_variance_risk_parity.annualized_vol)
str5 = """Fund holding cumulative return path: r = %f, $\sigma$ = %f. """ % \
       (equity_fund_portfolio_holdings.annualized_return,
        equity_fund_portfolio_holdings.annualized_vol)
# plt.title('%s to %s %s Optimizer performance comparison' % (equity_fund_portfolio_min_variance.daily_cum_log_return.index[0].date(),
#                                                             equity_fund_portfolio_min_variance.daily_cum_log_return.index[-1].date(),
#                                                              fund_name))

daily_cum_log_return.plot(legend=True, label=str1)
equity_fund_portfolio_min_variance.daily_cum_log_return.plot(legend=True, label=str2)
# equity_fund_portfolio_log_barrier.daily_cum_log_return.plot(legend=True, label=str3)
# equity_fund_portfolio_min_variance_risk_parity.daily_cum_log_return.plot(legend=True, label=str4)
equity_fund_portfolio_holdings.daily_cum_log_return.plot(legend=True, label=str5)
plt.ylabel('Cumulative Return', fontsize=18)
plt.xlabel('Time', fontsize=18)
matplotlib.rcParams.update({'font.size': 18})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc=4)
plt.show()



