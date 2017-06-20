# Black-Litterman model scratch @ Ricequant 06/15/2017

import numpy as np
import rqdatac
import pandas as pd
import ptfopt as pf

rqdatac.init('ricequant', '8ricequant8')


def black_litterman_prep(order_book_ids, start_date, investors_view, investors_view_indicate_M,
                         investors_view_uncertainty=None, asset_type=None, market_weight=None,
                         risk_free_rate_tenor=None, risk_aversion_coefficient=None, cov_uncertainty=None):

    risk_free_rate_dict = ['0S', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y',
                           '9Y', '10Y', '15Y', '20Y', '30Y', '40Y', '50Y']

    if market_weight is None:
        market_weight = [1/len(order_book_ids)] * len(order_book_ids)
    if risk_aversion_coefficient is None:
        risk_aversion_coefficient = 1
    if cov_uncertainty is None:
        cov_uncertainty = 0.025

    if asset_type is None:
        asset_type = "stock"
    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    clean_period_prices, reset_start_date = (pf.data_process(order_book_ids, asset_type, start_date)[i]
                                             for i in [0, 2])
    if risk_free_rate_tenor is None:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor='0S', country='cn')
    elif risk_free_rate_tenor in risk_free_rate_dict:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor=risk_free_rate_tenor,
                                                 country='cn')

    risk_free_rate['Daily'] = pd.Series(np.power(1 + risk_free_rate['0S'], 1 / 365) - 1, index=risk_free_rate.index)
    clean_period_prices_pct_change = clean_period_prices.pct_change()
    clean_period_excess_return = clean_period_prices_pct_change.subtract(risk_free_rate['Daily'], axis=0)

    equilibrium_return = np.multiply(np.dot(clean_period_excess_return[1:].cov(), market_weight),
                                     risk_aversion_coefficient)
    clean_period_excess_return_cov = clean_period_excess_return[1:].cov()

    if investors_view_uncertainty is None:
        Omeg_diag = list()
        for i in range(investors_view_indicate_M.shape[0]):
            temp = np.dot(np.dot(investors_view_indicate_M[i, :], clean_period_excess_return_cov),
                          investors_view_indicate_M[i, :].transpose()) * cov_uncertainty
            Omeg_diag.append(temp.item(0))
        investors_view_uncertainty = np.diag(Omeg_diag)

    combined_return_covar_M = np.linalg.inv(np.linalg.inv(np.multiply(cov_uncertainty, clean_period_excess_return_cov))
                                            + np.dot(np.dot(investors_view_indicate_M.transpose(),
                                                            np.linalg.inv(investors_view_uncertainty)),
                                                     investors_view_indicate_M))
    temp = np.dot(np.linalg.inv(np.multiply(cov_uncertainty, clean_period_excess_return_cov)),
                  equilibrium_return) + np.dot(np.dot(investors_view_indicate_M.transpose(),
                                                      np.linalg.inv(investors_view_uncertainty)), investors_view)
    combined_return_mean = np.dot(temp, combined_return_covar_M)
    return combined_return_mean, combined_return_covar_M

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']

investors_view = [0.02, 0.01]
investors_view_indicate_M = np.matrix([[0, -1, 0, 1, 0],
                                       [-1, 0, 1, 0, 0]])

res = black_litterman_prep(equity_funds_list, "2015-01-01", investors_view, investors_view_indicate_M, asset_type='fund')

print(res[0])
print(res[1])
