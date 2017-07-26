
order_book_ids = ['161826',
                  '150134',
                  '000404',
                  '550009',
                  '161116',
                  '118001',
                  '540006',
                  '000309']


rebalancing_date= '2014-01-01'
asset_type = 'fund'
#method = 'risk_parity'
#method = 'min_variance'
method = 'mean_variance'

bnds={'full_list': (0, 0.3) }
#bnds = None
cons = None

expected_return = None
risk_aversion_coef=1
window = 132
# benchmark = 'equal_weight'
benchmark = '000300.XSHG'
cov_shrinkage = True

from optimizer_for_engineer.back_test.portfolio_optimize import *



res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window= window,
                       bnds=bnds, cons=cons, cov_shrinkage = True,
                       benchmark = benchmark, industry_matching= True,
                       expected_return= None, risk_aversion_coef=1)


# res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window= 132,
#                        bnds=bnds, cons=cons, cov_shrinkage = True,
#                        benchmark = benchmark,
#                        expected_return= expected_return, risk_aversion_coef=risk_aversion_coef)

