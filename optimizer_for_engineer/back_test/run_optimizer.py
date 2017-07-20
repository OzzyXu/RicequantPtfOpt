
order_book_ids = ['161826',
                  '150134',
                  '000404',
                  '550009',
                  '161116',
                  '118001',
                  '540006',
                  '000309']


rebalancing_date= '2014-01-01'

end_date = '2017-05-01'
asset_type = 'fund'
method = 'risk_parity'

method= 'risk_parity'
#current_weight = None
rebalancing_frequency=66
window=132
bnds=None
cons=None
cov_shrinkage=True
expected_return=None
expected_return_cov=None
risk_aversion_coef=1
res_options='weight'

benchmark = 'equal_weight'






res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window= 132,
                       bnds=None, cons=None, cov_shrinkage = True,
                       benchmark = 'equal_weight',
                       expected_return= 'empirical_mean', risk_aversion_coef=1)

