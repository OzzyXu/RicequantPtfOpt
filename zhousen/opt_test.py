########################
# Created by ZS on 6/11
########################



# initialize rqdatac and import necessary packages


import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")

import numpy as np
import datetime as dt
import itertools
import pandas as pd
from math import *

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# create a variable called 'today' to send as default input


today_date = dt.date.today().strftime("%Y-%m-%d")
#print(before_date)







###############

# # get daily risk free rate
# risk_free_rate_tenor = None
#
# if risk_free_rate_tenor is None:
#     risk_free_rate = rqdatac.get_yield_curve(start_date, end_date, tenor='0S', country='cn')
# elif risk_free_rate_tenor in risk_free_rate_dict:
#     risk_free_rate = rqdatac.get_yield_curve(start_date, end_date, tenor=risk_free_rate_tenor, country='cn')
#
# risk_free_rate['Daily'] = pd.Series(np.power(1 + risk_free_rate['0S'], 1 / 365) - 1, index=risk_free_rate.index)


avg_risk_free_rate = np.mean(risk_free_rate['Daily'])

######### test and generate tables and plots


a1 = datetime.datetime(2017, 6, 22, 10, 40, 25, 143102) - datetime.datetime(2017, 6, 22, 10, 40, 28, 697069)
a1
a1.days
a1.seconds
a1.microseconds


a2 = datetime.datetime(2016, 6, 22, 10, 41, 28, 697069) - datetime.datetime(2017, 6, 22, 10, 40, 25, 143102)


a2
a2.days
a2.seconds

a3 = datetime.datetime(2016, 6, 22, 10, 40, 26, 0) - datetime.datetime(2017, 6, 22, 10, 40, 25, 1)

a3


start_date = '2017-1-1'
end_date = '2017-06-25'
end_date = today_date,

def optimizer_test(equity_funds_list, start_date,
                   end_date = dt.date.today().strftime("%Y/%m/%d"), frequency = 60,):

    # according to start_date to work out testing time_frame

    trading_date_s = get_previous_trading_date(start_date)
    trading_date_e = get_previous_trading_date(end_date)

    trading_dates = get_trading_dates(trading_date_s, trading_date_e)
    time_len = len(trading_dates)

    count = floor(time_len/frequency)


    #for i in range(frequency + 1):
    #    start_date = get_previous_trading_date(start_date)

    time_frame = {}
    #time_frame[0] = start_date
    for i in range(0, count+1):
        time_frame[i] = trading_dates[i*frequency]

  #  time_frame[count] = trading_dates[-1]


    # get time_frame fund_list acc_net_value



#    period_anv = {}
 #   for i in range(0, count+1):
  #      period_anv[i] = fund.get_nav(fund_list, start_date=second_period_s, end_date=seventh_period_e, fields='acc_net_value')

    opt_res = {}
    for i in range(0, count+1):
        opt_res[i] = optimizer(equity_funds_list, start_date=time_frame[i],  asset_type='fund', method='all')

    weights = {}

    methods = ['risk_parity', 'min_variance']


    daily_methods_a_r = {}

    for j in methods:
        daily_arithmetic_return = []

        for i in range(0, count):
            weights[i] = opt_res[i][0]
            period_prices_pct_change = opt_res[i+1][1]
            c_m = opt_res[i][2]


            period_daily_return_pct_change = pd.DataFrame(period_prices_pct_change)[1:]

            corresponding_data_in_weights = period_daily_return_pct_change.iloc[:,
                                            [x in weights[i].index for x in period_daily_return_pct_change.columns]]

            weighted_sum = corresponding_data_in_weights.multiply(weights[i][j]).sum(axis=1)
            daily_arithmetic_return.append(weighted_sum)
            print(daily_arithmetic_return)

        daily_methods_a_r[j] = daily_arithmetic_return

#    return daily_methods_a_r

 #   optimizer_test(equity_funds_list, start_date)


    print(daily_methods_a_r)


    annualized_vol = {}
    annualized_return = {}

    for j in methods:
        #print(j)
        #print(daily_methods_a_r)
        temp = np.log(daily_methods_a_r[j][0] + 1)

        annualized_vol[j] = sqrt(244) * temp.std()
        days_count = len(daily_methods_a_r[j])
        daily_cum_log_return = temp.cumsum()
        annualized_return[j] = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1

    #fig1 = plt.figure()


    str1 = """Fund NAV cumulative return path: r = %f, $\sigma$ = %f. """ %(annualized_return[j], annualized_vol[j])
    #str2 = """Minimum variance portfolio cumulative return path: r = %f, $\sigma$ = %f.""" % \
    #       (equity_fund_portfolio_min_variance.annualized_return, equity_fund_portfolio_min_variance.annualized_vol)

    plt.figure(1)
    p1 = daily_cum_log_return.plot(legend=True, label=str1)


    plt.figure(2)
    p2 = daily_cum_log_return.plot(legend=True, label=str1)

    return weights, annualized_return, annualized_vol, p1




optimizer_test(equity_funds_list, start_date, end_date , frequency = 60)



#########




    plt.ylabel('Cumulative Return', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc=4)
    plt.show()

    new_daily_arithmetic_return = period_daily_return_pct_change.multiply(weights).sum(axis=1)





period_daily_return_pct_change.iloc[:,[x in weights[i].index for x in period_daily_return_pct_change.columns]].head().multiply(weights.iloc[:,1])

period_daily_return_pct_change['002901'].head()*3.204039e-01


        daily_cum_log_return = np.log(period_daily_return_pct_change + 1).cumsum()

        temp = np.log(period_daily_return_pct_change + 1)
        annualized_vol = math.sqrt(244) * temp.std()
        days_count = (period_daily_return_pct_change.index[-1] - period_daily_return_pct_change.index[0]).days
        annualized_return = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1





if self.daily_arithmetic_return is None:
    self.daily_arithmetic_return = new_daily_arithmetic_return
else:
    self.daily_arithmetic_return = self.daily_arithmetic_return.append(new_daily_arithmetic_return)

new_daily_cum_log_return = np.log(new_daily_arithmetic_return + 1).cumsum()
if self.daily_cum_log_return is None:
    self.daily_cum_log_return = new_daily_cum_log_return
else:
    self.daily_cum_log_return = self.daily_cum_log_return.append(
        new_daily_cum_log_return + self.daily_cum_log_return[-1])

temp = np.log(self.daily_arithmetic_return + 1)
self.annualized_vol = sqrt(244) * temp.std()
days_count = (self.daily_arithmetic_return.index[-1] - self.daily_arithmetic_return.index[0]).days
self.annualized_return = (self.daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1




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







#########









########## draft


d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),
     'three': pd.Series([5., 6., 7., 8], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)



df[df['one'] == 1]
['two']


itertools.combinations([1,2,3], 2)


import itertools

stuff = [1, 2, 3]
for L in range(0, len(stuff)+1):
  for subset in itertools.combinations(stuff, L):
    print(subset)


from itertools import chain, combinations
def all_subsets(ss):
  return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

for subset in all_subsets(stuff):
  print(subset)


letters=pd.Series(('A', 'B', 'C', 'D'))
numbers=pd.Series((1, 2, 3, 4))
keys=('Letters', 'Numbers')
df=pd.concat((letters, numbers), axis=1, keys=keys)
df[df.Letters=='C'].Letters.item()


mydict =fund_pool
dict(list(mydict.items())[0:1])


