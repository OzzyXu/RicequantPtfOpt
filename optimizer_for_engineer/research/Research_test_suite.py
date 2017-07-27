############################################################################################
# fix the following
# first test fund
###################


from optimizer_for_engineer.research.portfolio_optimize import *

order_book_ids = ['161826',
                  '150134',
                  '000404',
                  '550009',
                  '161116',
                  '118001',
                  '540006',
                  '000309']
start_date = '2014-01-01'
end_date = '2015-05-01'
method = 'all'

asset_type = 'fund'


#############################
# case 1


bnds = None
cons = None
cov_shrinkage0 = False
benchmark = 'equal_weight'
industry_matching = False
res_options = 'weight'

#######
# run


res = portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method=method,
                         rebalancing_frequency=66, window=132, bnds=bnds, cons=cons,
                         cov_shrinkage=cov_shrinkage0, benchmark=benchmark,
                         industry_matching=industry_matching, expected_return=None,
                         risk_aversion_coef=1, res_options=res_options)

res
# {'optimizer_status': {0:                                      Opt Res Message
#   risk_parity    Optimization terminated successfully.
#   min_variance   Optimization terminated successfully.
#   mean_variance  Optimization terminated successfully.,
#   1:                                      Opt Res Message
#   risk_parity    Optimization terminated successfully.
#   min_variance   Optimization terminated successfully.
#   mean_variance  Optimization terminated successfully.,
#   2:                                      Opt Res Message
#   risk_parity    Optimization terminated successfully.
#   min_variance   Optimization terminated successfully.
#   mean_variance  Optimization terminated successfully.,
#   3:                                      Opt Res Message
#   risk_parity    Optimization terminated successfully.
#   min_variance   Optimization terminated successfully.
#   mean_variance  Optimization terminated successfully.,
#   4:                                      Opt Res Message
#   risk_parity    Optimization terminated successfully.
#   min_variance   Optimization terminated successfully.
#   mean_variance  Optimization terminated successfully.},
#  'weights': {0:         risk_parity  min_variance  mean_variance  equal_weight
#   161116     0.227612      0.330273   2.829797e-16           0.2
#   118001     0.194722      0.262868   0.000000e+00           0.2
#   540006     0.100398      0.000000   6.938894e-17           0.2
#   550009     0.111376      0.025755   1.000000e+00           0.2
#   161826     0.365892      0.381105   1.854244e-16           0.2,
#   1:         risk_parity  min_variance  mean_variance  equal_weight
#   118001     0.241221      0.295112   1.359740e-16      0.166667
#   161116     0.262066      0.323614   6.366502e-17      0.166667
#   000404     0.082592      0.000000   1.000000e+00      0.166667
#   540006     0.083386      0.049548   1.236170e-16      0.166667
#   550009     0.083309      0.038967   0.000000e+00      0.166667
#   161826     0.247426      0.292759   1.313953e-17      0.166667,
#   2:         risk_parity  min_variance  mean_variance  equal_weight
#   118001     0.181995      0.272308   1.137577e-16      0.142857
#   161116     0.301524      0.368558   1.000000e+00      0.142857
#   000309     0.084710      0.043355   0.000000e+00      0.142857
#   000404     0.061860      0.000000   0.000000e+00      0.142857
#   540006     0.077926      0.015932   0.000000e+00      0.142857
#   550009     0.075578      0.000000   7.735062e-17      0.142857
#   161826     0.216407      0.299846   1.942890e-16      0.142857,
#   3:         risk_parity  min_variance  mean_variance  equal_weight
#   118001     0.137901      0.218519   3.701222e-16      0.142857
#   161116     0.365985      0.361140   2.361740e-16      0.142857
#   000309     0.086439      0.065154   1.277806e-16      0.142857
#   000404     0.073231      0.000000   3.183468e-16      0.142857
#   540006     0.087258      0.078722   0.000000e+00      0.142857
#   550009     0.081491      0.044363   1.000000e+00      0.142857
#   161826     0.167693      0.232103   0.000000e+00      0.142857,
#   4:         risk_parity  min_variance  mean_variance  equal_weight
#   118001     0.197390      0.323944   5.548172e-16      0.142857
#   161116     0.369676      0.483940   3.491936e-16      0.142857
#   000309     0.097138      0.047605   0.000000e+00      0.142857
#   000404     0.102319      0.057157   2.354703e-17      0.142857
#   540006     0.080828      0.000000   1.000000e+00      0.142857
#   550009     0.071778      0.083048   0.000000e+00      0.142857
#   161826     0.080871      0.004306   1.963831e-16      0.142857}}




#############################
# case 2

bnds = {'full_list': (0, 0.4)}
# bnds = None
cons = {'BondIndex': (0.3, 0.5)}
cov_shrinkage0 = True
benchmark = 'equal_weight'
industry_matching = False
res_options = 'all'
#######
# run

res = portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method=method,
                         rebalancing_frequency=66, window=132, bnds=bnds, cons=cons,
                         cov_shrinkage=cov_shrinkage0, benchmark=benchmark,
                         industry_matching=industry_matching, expected_return=None,
                         risk_aversion_coef=1, res_options=res_options)

res['weights']

# {0:         risk_parity  min_variance  mean_variance  equal_weight
#  550009     0.113017      0.034986   4.000000e-01           0.2
#  161826     0.365387      0.400000   3.000000e-01           0.2
#  161116     0.223202      0.313117   0.000000e+00           0.2
#  118001     0.195442      0.251898   1.387779e-16           0.2
#  540006     0.102952      0.000000   3.000000e-01           0.2,
#  1:         risk_parity  min_variance  mean_variance  equal_weight
#  550009     0.079913      0.022687   3.000000e-01      0.166667
#  161826     0.300000      0.398469   3.000000e-01      0.166667
#  540006     0.077992      0.027277   5.533768e-16      0.166667
#  118001     0.220697      0.260252   5.394990e-16      0.166667
#  161116     0.243855      0.291315   2.602085e-16      0.166667
#  000404     0.077543      0.000000   4.000000e-01      0.166667,
#  2:         risk_parity  min_variance  mean_variance  equal_weight
#  550009     0.068018  7.704050e-18   0.000000e+00      0.142857
#  161826     0.300000  3.413988e-01   4.000000e-01      0.142857
#  540006     0.070087  1.146426e-02   1.295216e-17      0.142857
#  118001     0.159325  2.512475e-01   0.000000e+00      0.142857
#  161116     0.269682  3.531244e-01   4.000000e-01      0.142857
#  000404     0.056603  1.558574e-19   2.000000e-01      0.142857
#  000309     0.076285  4.276497e-02   0.000000e+00      0.142857,
#  3:         risk_parity  min_variance  mean_variance  equal_weight
#  550009     0.094175      0.029639       0.400000      0.142857
#  161826     0.300000      0.300000       0.400000      0.142857
#  540006     0.102767      0.063383       0.000000      0.142857
#  118001     0.164101      0.196960       0.000000      0.142857
#  161116     0.151319      0.355305       0.000000      0.142857
#  000404     0.086823      0.000000       0.123953      0.142857
#  000309     0.100815      0.054712       0.076047      0.142857,
#  4:         risk_parity  min_variance  mean_variance  equal_weight
#  550009     0.047027  1.639160e-17   3.000000e-01      0.142857
#  161826     0.300000  3.000000e-01   3.000000e-01      0.142857
#  540006     0.072652  0.000000e+00   4.000000e-01      0.142857
#  118001     0.148443  2.118680e-01   1.331400e-16      0.142857
#  161116     0.149161  4.000000e-01   2.712674e-16      0.142857
#  000404     0.149361  5.539650e-02   0.000000e+00      0.142857
#  000309     0.133357  3.273549e-02   2.359224e-16      0.142857}







############################################################################################
# fix the following
# next test stock
#######################

from optimizer_for_engineer.research.portfolio_optimize import *

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


start_date = '2014-01-01'
end_date = '2015-05-01'
method = 'all'

asset_type = 'stock'

#############################
# case 1

bnds = {'full_list': (0, 0.4)}
cons = None
cov_shrinkage0 = True
benchmark = '000300.XSHG'
industry_matching = True
res_options = 'all'
#######
# run

res = portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method=method,
                         rebalancing_frequency=66, window=132, bnds=bnds, cons=cons,
                         cov_shrinkage=cov_shrinkage0, benchmark=benchmark,
                         industry_matching=industry_matching, expected_return=None,
                         risk_aversion_coef=1, res_options=res_options)

res['weights']

# {0:              risk_parity  min_variance  mean_variance
#  601006.XSHG     0.075862  8.206835e-02   2.339398e-01
#  600519.XSHG     0.088298  1.177393e-01   1.811206e-17
#  601988.XSHG     0.082978  1.068068e-01   8.760977e-17
#  601939.XSHG     0.074254  7.400261e-02   9.628455e-17
#  600018.XSHG     0.039725  0.000000e+00   3.261584e-01
#  601633.XSHG     0.046072  2.254338e-18   2.552979e-01
#  600028.XSHG     0.069261  5.406791e-02   7.463628e-17
#  601288.XSHG     0.071410  6.628565e-02   1.842369e-16
#  601818.XSHG     0.057898  1.591298e-18   7.344280e-17
#  601857.XSHG     0.107343  1.626914e-01   1.505323e-16
#  601398.XSHG     0.102297  1.517341e-01   2.551601e-17,
#  1:              risk_parity  min_variance  mean_variance
#  601006.XSHG     0.052282  4.959040e-02   0.000000e+00
#  600519.XSHG     0.049429  3.768686e-02   2.419962e-01
#  601988.XSHG     0.066101  8.688526e-02   2.915503e-02
#  601939.XSHG     0.058516  6.707957e-02   5.345847e-17
#  600018.XSHG     0.031478  0.000000e+00   0.000000e+00
#  601633.XSHG     0.027324  4.197965e-18   0.000000e+00
#  600028.XSHG     0.041576  6.690507e-18   2.419962e-01
#  601288.XSHG     0.071019  9.626090e-02   3.791102e-02
#  601818.XSHG     0.054417  5.965041e-02   0.000000e+00
#  601857.XSHG     0.071709  9.552563e-02   5.393207e-02
#  601398.XSHG     0.081140  1.123115e-01   3.017288e-18,
#  2:              risk_parity  min_variance  mean_variance
#  601006.XSHG     0.051435  6.002402e-02   1.581483e-17
#  600519.XSHG     0.034455  6.792139e-04   2.368582e-01
#  601988.XSHG     0.062721  8.031176e-02   7.500915e-02
#  601939.XSHG     0.075237  9.418354e-02   0.000000e+00
#  600018.XSHG     0.033767  2.054418e-18   0.000000e+00
#  601633.XSHG     0.028928  0.000000e+00   5.992942e-18
#  600028.XSHG     0.038040  2.396703e-02   2.368582e-01
#  601288.XSHG     0.071124  8.950787e-02   8.217671e-17
#  601818.XSHG     0.051931  6.298904e-02   0.000000e+00
#  601857.XSHG     0.074905  9.155259e-02   9.600485e-18
#  601398.XSHG     0.069605  8.893055e-02   4.341997e-02,
#  3:              risk_parity  min_variance  mean_variance
#  601006.XSHG     0.043903      0.047827   2.291085e-01
#  600519.XSHG     0.037728      0.033623   0.000000e+00
#  601988.XSHG     0.060405      0.065689   0.000000e+00
#  601939.XSHG     0.066020      0.069417   4.302724e-17
#  600018.XSHG     0.046155      0.049689   2.291085e-01
#  601633.XSHG     0.029001      0.000000   1.919998e-16
#  600028.XSHG     0.046127      0.050875   0.000000e+00
#  601288.XSHG     0.059241      0.065353   0.000000e+00
#  601818.XSHG     0.042366      0.047574   1.145542e-01
#  601857.XSHG     0.077620      0.073955   2.494818e-17
#  601398.XSHG     0.064206      0.068769   0.000000e+00,
#  4:              risk_parity  min_variance  mean_variance
#  601006.XSHG     0.052805  6.485959e-02   2.226871e-17
#  600519.XSHG     0.079546  2.130446e-01   0.000000e+00
#  601988.XSHG     0.039051  1.963359e-18   2.130446e-01
#  601939.XSHG     0.039014  1.905613e-18   1.956773e-18
#  600018.XSHG     0.057834  9.828907e-02   0.000000e+00
#  601633.XSHG     0.049457  5.053989e-02   5.115754e-02
#  600028.XSHG     0.045371  2.614275e-02   7.735268e-17
#  601288.XSHG     0.041255  1.424867e-02   1.407445e-17
#  601818.XSHG     0.040834  2.728571e-02   2.130446e-01
#  601857.XSHG     0.042382  2.834557e-03   5.536477e-02
#  601398.XSHG     0.045064  3.536667e-02   7.524917e-17}
