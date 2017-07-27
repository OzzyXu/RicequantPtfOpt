#############################
# fix the following
# first test fund

from optimizer_for_engineer.back_test.portfolio_optimize import *

order_book_ids = ['161826',
                  '150134',
                  '000404',
                  '550009',
                  '161116',
                  '118001',
                  '540006',
                  '000309']
rebalancing_date = '2014-01-01'
window = 132
asset_type = 'fund'

#############################
# case 1


method = 'risk_parity'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run


res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':         risk_parity           status
#  118001     0.195725                0
#  161116     0.227632                0
#  161826     0.364558                0
#  540006     0.100680                0
#  550009     0.111406                0
#  150134     0.000000          无相关基金数据
#  000404     0.000000  基金发行时间少于132个交易日
#  000309     0.000000  基金发行时间少于132个交易日}




#############################
# case 2


method = 'min_variance'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':         min_variance           status
#  118001  2.640510e-01                0
#  161116  3.301337e-01                0
#  161826  3.805948e-01                0
#  540006  2.775558e-17                0
#  550009  2.522054e-02                0
#  150134  0.000000e+00          无相关基金数据
#  000404  0.000000e+00  基金发行时间少于132个交易日
#  000309  0.000000e+00  基金发行时间少于132个交易日}




#############################
# case 3

method = 'mean_variance'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':         mean_variance           status
#  118001   8.688870e-17                0
#  161116   6.782902e-18                0
#  161826   0.000000e+00                0
#  540006   0.000000e+00                0
#  550009   1.000000e+00                0
#  150134   0.000000e+00          无相关基金数据
#  000404   0.000000e+00  基金发行时间少于132个交易日
#  000309   0.000000e+00  基金发行时间少于132个交易日}



#############################
# case 4


method = 'risk_parity'
bnds = {'full_list': (0, 0.3)}
cons = None
cov_shrinkage0 = True
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# Out[5]:
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':         risk_parity           status
#  540006     0.122316                0
#  118001     0.230197                0
#  550009     0.132063                0
#  161116     0.248888                0
#  161826     0.266537                0
#  150134     0.000000          无相关基金数据
#  000404     0.000000  基金发行时间少于132个交易日
#  000309     0.000000  基金发行时间少于132个交易日}






########################################
# fix the following
# next test stock

from optimizer_for_engineer.back_test.portfolio_optimize import *

order_book_ids = ['601857.XSHG',
 '601398.XSHG',
 '601939.XSHG',
 '601288.XSHG',
 '601988.XSHG',
 '600028.XSHG',
 '600519.XSHG',
 '601633.XSHG',
 '601818.XSHG',
 '600018.XSHG',
 '601006.XSHG']


rebalancing_date = '2014-01-01'
window = 132
asset_type = 'stock'


#############################
# case 1




method = 'risk_parity'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run


res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':              risk_parity  status
#  601398.XSHG     0.122549       0
#  601288.XSHG     0.084405       0
#  601633.XSHG     0.059141       0
#  600018.XSHG     0.053432       0
#  601006.XSHG     0.095582       0
#  601939.XSHG     0.088816       0
#  600028.XSHG     0.084197       0
#  600519.XSHG     0.113663       0
#  601818.XSHG     0.067695       0
#  601988.XSHG     0.097841       0
#  601857.XSHG     0.132679       0}




#############################
# case 2


method = 'min_variance'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':              min_variance  status
#  601398.XSHG  1.719759e-01       0
#  601288.XSHG  7.600828e-02       0
#  601633.XSHG  7.155734e-18       0
#  600018.XSHG  1.502215e-02       0
#  601006.XSHG  1.114318e-01       0
#  601939.XSHG  8.815055e-02       0
#  600028.XSHG  7.294822e-02       0
#  600519.XSHG  1.546595e-01       0
#  601818.XSHG  6.835895e-17       0
#  601988.XSHG  1.192608e-01       0
#  601857.XSHG  1.905428e-01       0}





#############################
# case 3

method = 'mean_variance'
bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':              mean_variance  status
#  601398.XSHG   0.000000e+00       0
#  601288.XSHG   0.000000e+00       0
#  601633.XSHG   0.000000e+00       0
#  600018.XSHG   1.000000e+00       0
#  601006.XSHG   0.000000e+00       0
#  601939.XSHG   3.479814e-17       0
#  600028.XSHG   0.000000e+00       0
#  600519.XSHG   0.000000e+00       0
#  601818.XSHG   0.000000e+00       0
#  601988.XSHG   4.879765e-18       0
#  601857.XSHG   0.000000e+00       0}




#############################
# case 4


method = 'risk_parity'
bnds = {'full_list': (0, 0.3)}
cons = {'银行': (0.3, 0.5)}
cov_shrinkage0 = True
benchmark = 'equal_weight'
industry_matching = False

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':              risk_parity  status
#  601398.XSHG     0.125254       0
#  601288.XSHG     0.087782       0
#  601633.XSHG     0.056903       0
#  600018.XSHG     0.049167       0
#  601006.XSHG     0.092589       0
#  601939.XSHG     0.091097       0
#  600028.XSHG     0.085063       0
#  600519.XSHG     0.106288       0
#  601818.XSHG     0.071203       0
#  601988.XSHG     0.102346       0
#  601857.XSHG     0.132307       0}



#############################
# case 5


method = 'risk_parity'
bnds = {'full_list': (0, 0.3)}
cons = {'银行': (0.3, 0.5)}
cov_shrinkage0 = True
benchmark = '000300.XSHG'
industry_matching = True

#######
# run

res = portfolio_optimize(order_book_ids, rebalancing_date, asset_type, method, window=window,
                         bnds=bnds, cons=cons, cov_shrinkage=cov_shrinkage0,
                         benchmark=benchmark, industry_matching=industry_matching,
                         expected_return=None, risk_aversion_coef=1)

res
# {'industry_matching_weight': 002299.XSHE    0.000506641
#  002069.XSHE    0.000599234
#  600108.XSHG    0.000902954
#  600598.XSHG     0.00116123
#  000876.XSHE     0.00143819
#  002385.XSHE     0.00151679
#  601118.XSHG     0.00168699
#  000778.XSHE    0.000893608
#  000709.XSHE     0.00122659
#  600010.XSHG      0.0019921
#  600019.XSHG     0.00389104
#  002155.XSHE       0.000454
#  600259.XSHG    0.000559616
#  600219.XSHG    0.000582016
#  000758.XSHE    0.000611379
#  000960.XSHE    0.000710121
#  000878.XSHE    0.000712536
#  601168.XSHG    0.000741849
#  000060.XSHE    0.000748255
#  600516.XSHG    0.000755621
#  000970.XSHE    0.000789949
#  000630.XSHE    0.000822716
#  600497.XSHG    0.000902452
#  600549.XSHG    0.000946911
#  000831.XSHE      0.0012492
#  601958.XSHG      0.0013511
#  600547.XSHG     0.00141781
#  600489.XSHG     0.00145343
#  603993.XSHG     0.00190276
#  601600.XSHG     0.00265584
#                    ...
#  000656.XSHE    0.000540661
#  600376.XSHG    0.000651342
#  600895.XSHG    0.000671749
#  000402.XSHE    0.000914384
#  002146.XSHE     0.00108699
#  600208.XSHG     0.00115677
#  000046.XSHE     0.00118184
#  600649.XSHG     0.00143734
#  600340.XSHG     0.00154568
#  600383.XSHG     0.00172518
#  600663.XSHG     0.00183274
#  600648.XSHG     0.00188157
#  000024.XSHE     0.00206108
#  000069.XSHE     0.00222588
#  600048.XSHG     0.00340121
#  000002.XSHE     0.00510802
#  600859.XSHG     0.00048538
#  600694.XSHG    0.000489927
#  600655.XSHG    0.000644197
#  000061.XSHE    0.000825254
#  600058.XSHG    0.000839501
#  600415.XSHG    0.000922713
#  600827.XSHG    0.000981925
#  601933.XSHG     0.00124903
#  002344.XSHE     0.00134421
#  002024.XSHE     0.00385058
#  601888.XSHG     0.00196781
#  000839.XSHE    0.000565991
#  600811.XSHG    0.000605534
#  000009.XSHE    0.000684633
#  Name: percent, dtype: object,
#  'optimizer_status': 'Optimization terminated successfully.',
#  'total_weight':              risk_parity  status
#  601398.XSHG     0.102132       0
#  601288.XSHG     0.071577       0
#  601633.XSHG     0.046398       0
#  600018.XSHG     0.040091       0
#  601006.XSHG     0.075497       0
#  601939.XSHG     0.074280       0
#  600028.XSHG     0.069360       0
#  600519.XSHG     0.086667       0
#  601818.XSHG     0.058059       0
#  601988.XSHG     0.083452       0
#  601857.XSHG     0.107883       0}





