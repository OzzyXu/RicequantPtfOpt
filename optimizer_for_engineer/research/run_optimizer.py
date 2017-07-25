#
# order_book_ids = ['161826',
#                   '150134',
#                   '000404',
#                   '550009',
#                   '161116',
#                   '118001',
#                   '540006',
#                   '000309']
#
#
# start_date= '2014-01-01'
# end_date = '2015-05-01'
# asset_type = 'fund'
# method='all'
# benchmark = 'equal_weight'
#
# res_options = 'weight'
#
# res_options = 'all'
#
#
# from optimizer_for_engineer.research.portfolio_optimize import *
#
# res = portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = method,
#                          rebalancing_frequency = 66, window= 132, bnds=None, cons = cons,
#                          cov_shrinkage = True, benchmark = benchmark,
#                          industry_matching = False, expected_return= None,
#                          risk_aversion_coef=1, res_options = res_options)
#



order_book_ids = ['601857.XSHG',
 '601398.XSHG',
 '601939.XSHG',
 '601288.XSHG',
 '601988.XSHG',
 '600028.XSHG',
 '601628.XSHG',
 '601318.XSHG',
 '601088.XSHG',
 '601328.XSHG',
 '600036.XSHG',
 '600016.XSHG',
 '601166.XSHG',
 '601998.XSHG',
 '600000.XSHG',
 '601601.XSHG',
 '600104.XSHG',
 '600030.XSHG',
 '600519.XSHG',
 '601633.XSHG',
 '601818.XSHG',
 '600018.XSHG',
 '601006.XSHG',
 '600837.XSHG',
 '600900.XSHG',
 '000895.XSHE',
 '000001.XSHE',
 '601808.XSHG',
 '000651.XSHE',
 '601668.XSHG',
 '002415.XSHE',
 '600585.XSHG',
 '002594.XSHE',
 '000002.XSHE',
 '601989.XSHG',
 '600887.XSHG',
 '600015.XSHG',
 '000776.XSHE',
 '601336.XSHG',
 '600011.XSHG',
 '000538.XSHE',
 '601766.XSHG',
 '600050.XSHG',
 '600019.XSHG',
 '002024.XSHE',
 '600600.XSHG',
 '601169.XSHG',
 '601800.XSHG',
 '601898.XSHG',
 '000858.XSHE',
 '600999.XSHG',
 '600048.XSHG',
 '601186.XSHG',
 '601390.XSHG',
 '601991.XSHG',
 '600111.XSHG',
 '002241.XSHE',
 '000625.XSHE',
 '600690.XSHG',
 '601238.XSHG',
 '601111.XSHG',
 '600276.XSHG',
 '601899.XSHG',
 '601688.XSHG',
 '600362.XSHG',
 '600031.XSHG',
 '601727.XSHG',
 '002353.XSHE',
 '002236.XSHE',
 '601600.XSHG',
 '600256.XSHG',
 '000063.XSHE',
 '600309.XSHG',
 '600535.XSHG',
 '002304.XSHE',
 '600196.XSHG',
 '600188.XSHG',
 '000157.XSHE',
 '600372.XSHG',
 '600637.XSHG',
 '600795.XSHG',
 '601607.XSHG',
 '600518.XSHG',
 '000039.XSHE',
 '601117.XSHG',
 '000069.XSHE',
 '000338.XSHE',
 '300070.XSHE',
 '601901.XSHG',
 '600703.XSHG',
 '600332.XSHG',
 '600115.XSHG',
 '600010.XSHG',
 '600583.XSHG',
 '601888.XSHG',
 '300027.XSHE',
 '601618.XSHG',
 '600150.XSHG',
 '603993.XSHG',
 '600688.XSHG']


start_date= '2014-01-01'
end_date = '2015-05-01'
asset_type = 'stock'
method='all'

benchmark = '000300.XSHG'
#benchmark = 'equal_weight'

#res_options = 'weight'

res_options = 'all'


cons = {'银行': (0.3, 0.5)}
#cons = None

from optimizer_for_engineer.research.portfolio_optimize import *
res = portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = method,
                         rebalancing_frequency = 66, window= 132, bnds=None, cons=cons,
                         cov_shrinkage = True, benchmark = benchmark,
                         industry_matching = True, expected_return= None,
                         risk_aversion_coef=1, res_options = res_options)



