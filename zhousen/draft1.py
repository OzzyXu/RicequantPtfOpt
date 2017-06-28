
import xuchuan.ptfopt as pt










fund_types
try_suite = list(fund_pool['Bond'])[0:4] + list(fund_pool['Stock'])[0:4]
try_suite


order_book_ids = try_suite
start_date = '2017-06-19'
asset_type = 'fund'
method = 'risk_parity'

method = 'all'
equity_funds_list = ['002832', '002901', ]



optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')








import ptfopt_zs as pt

stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = ['002832',
['002901',
 '002341',
 '003176',
 '003634',
 '002621',
 '000916',
 '001416']

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']





equity_funds_list = ['002832',
 '002901',
 '002341',
 '003176',
 '003634',
 '002621',
 '000916',
 '001416']

a1=optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')

a1[0]
a2 = a1[1].head()
a2.pct_change()

import ptfopt_zs as pt
pt.optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')



optimal_weight[0]


print(optimal_weight[0])






index_name = "000300.XSHG"
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

portfolio1 = rqdatac.index_weights(index_name, second_period_s)
equity_list1 = list(portfolio1.index)
portfolio2 = rqdatac.index_weights(index_name, third_period_s)
equity_list2 = list(portfolio2.index)
portfolio3 = rqdatac.index_weights(index_name, fourth_period_s)
equity_list3 = list(portfolio3.index)
portfolio4 = rqdatac.index_weights(index_name, fifth_period_s)
equity_list4 = list(portfolio4.index)
portfolio5 = rqdatac.index_weights(index_name, sixth_period_s)
equity_list5 = list(portfolio5.index)
portfolio6 = rqdatac.index_weights(index_name, seventh_period_s)
equity_list6 = list(portfolio6.index)

# Min variance optimizer
equity_fund_portfolio_min_variance = pt.TestPortfolio(equity_list1, 'stocks')
t = time.time()
equity_fund_portfolio_min_variance.data_preprocessing(equity_list1, first_period_s, first_period_e)
print(time.time()-t)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio1.loc[elimination_list])
start_time1 = time.time()
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
print('MV optimizer running time 1: %s' % (time.time()-start_time1))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res1 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res1.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20140701.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list2, second_period_s, second_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio2.loc[elimination_list])
start_time2 = time.time()
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
print('MV optimizer running time 2: %s' % (time.time()-start_time2))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res2 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res2.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20150101.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list3, third_period_s, third_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio3.loc[elimination_list])
start_time3 = time.time()
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
print('MV optimizer running time 3: %s' % (time.time()-start_time3))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res3 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res3.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20150701.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list4, fourth_period_s, fourth_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio4.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res4 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res4.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20160101.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list5, fifth_period_s, fifth_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio5.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res5 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res5.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20160701.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list6, sixth_period_s, sixth_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio6.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res6 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res6.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20170101.csv')

# Log barrier risk parity optimizer
equity_fund_portfolio_log_barrier = pt.TestPortfolio(equity_list1, 'stocks')
equity_fund_portfolio_log_barrier.data_preprocessing(equity_list1, first_period_s, first_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio1.loc[elimination_list])
start_time4 = time.time()
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
print('RP optimizer running time 1: %s' % (time.time()-start_time4))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res7 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res7.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20140701.csv')

equity_fund_portfolio_log_barrier.data_preprocessing(equity_list2, second_period_s, second_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio2.loc[elimination_list])
start_time5 = time.time()
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
print('RP optimizer running time 2: %s' % (time.time()-start_time5))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res8 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res8.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20150101.csv')

equity_fund_portfolio_log_barrier.data_preprocessing(equity_list3, third_period_s, third_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio3.loc[elimination_list])
start_time6 = time.time()
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
print('RP optimizer running time 3: %s' % (time.time()-start_time6))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res9 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res9.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20150701.csv')

equity_fund_portfolio_log_barrier.data_preprocessing(equity_list4, fourth_period_s, fourth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio4.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res10 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res10.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20160101.csv')

equity_fund_portfolio_log_barrier.data_preprocessing(equity_list5, fifth_period_s, fifth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio5.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res11 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res11.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20160701.csv')

equity_fund_portfolio_log_barrier.data_preprocessing(equity_list6, sixth_period_s, sixth_period_e)
elimination_list = equity_fund_portfolio_log_barrier.kickout_list+equity_fund_portfolio_log_barrier.st_list + \
                   equity_fund_portfolio_log_barrier.suspended_list
inherited_holdings_weights = list(portfolio6.loc[elimination_list])
optimal_weights = list(equity_fund_portfolio_log_barrier.log_barrier_risk_parity_optimizer())
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res12 = pd.Series(weights, index=(equity_fund_portfolio_log_barrier.clean_equity_list+elimination_list))
output_res12.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\RP20170101.csv')