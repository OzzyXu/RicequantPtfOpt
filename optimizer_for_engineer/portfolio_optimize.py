#

def portfolio_optimize(order_book_ids, start_date, end_date, asset_type, method = 'all',
                       rebalancing_frequency = 66, window= 132, bnds=None, cons=None,
                       cov_shrinkage = True, benchmark = 'equal_weight', industry_matching = False,
                       expected_return= 'empirical_mean', risk_aversion_coef=1, res_options = 'weight'):


