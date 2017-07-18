
###### import ######







from optimizer_for_engineer.portfolio_optimize import *



# 传入参数

# order_book_ids = ['000001.XSHE', '000024.XSHE']
order_book_ids = ['450006', '161211']


order_book_ids = ['519682',
 '166016',
 '000014',
 '161826',
 '150134',
 '160124',
 '166802',
 '000368',
 '160218']

start_date= '2014-01-01'

end_date = '2017-05-01'
asset_type = 'fund'
method='all'
current_weight = None
rebalancing_frequency=66
window=132
bnds=None
cons=None
cov_shrinkage=False
expected_return=None
expected_return_cov=None
risk_aversion_coefficient=1
res_options='weights'


try:
    input_validation(order_book_ids, start_date, end_date, asset_type, method ='risk_parity',
                     rebalancing_frequency = 66, window=132, bnds = None, cons = None, cov_shrinkage= False,
                     expected_return=None, expected_return_cov=None, risk_aversion_coefficient=1,
                     res_options='all')
except OptimizationError:
    print('请重新输入参数！')


###### TEST ######






portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method='all', current_weight = None, rebalancing_frequency=66,
                       window=132,
                       bnds=None, cons=None, cov_shrinkage=False, expected_return=None, expected_return_cov=None,
                       risk_aversion_coefficient=1, res_options='weights')




a = optimizer(order_book_ids, start_date, asset_type, method)

(order_book_ids, start_date, asset_type, method,






def portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = 'all',
                       rebalancing_frequency = 66, window= 132, bnds=None, cons=None,
                       cov_shrinkage = True, benchmark = 'equal_weight',
                       industry_matching = False, expected_return= 'empirical_mean',
                       risk_aversion_coef=1, res_options = 'weight'):