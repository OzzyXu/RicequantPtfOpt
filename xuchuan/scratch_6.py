# Black-Litterman model scratch @ Ricequant 06/15/2017

import numpy as np
import rqdatac
import pandas as pd
import xuchuan.ptfopt as pf
import scipy.optimize as sc_opt
from math import *

rqdatac.init('ricequant', '8ricequant8')


def black_litterman_prep(order_book_ids, start_date, investors_views, investors_views_indicate_M,
                         investors_views_uncertainty=None, asset_type=None, market_weight=None,
                         risk_free_rate_tenor=None, risk_aversion_coefficient=None, excess_return_cov_uncertainty=None,
                         confidence_of_views=None):

    risk_free_rate_dict = ['0S', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y',
                           '9Y', '10Y', '15Y', '20Y', '30Y', '40Y', '50Y']
    windows = 132
    if market_weight is None:
        market_weight = pd.DataFrame([1/len(order_book_ids)] * len(order_book_ids), index=order_book_ids)

    # Clean data
    if asset_type is None:
        asset_type = "fund"
    end_date = rqdatac.get_previous_trading_date(start_date)
    end_date = pd.to_datetime(end_date)
    clean_period_prices, reset_start_date = (pf.data_process(order_book_ids, asset_type, start_date, windows)[i]
                                             for i in [0, 2])
    
    if excess_return_cov_uncertainty is None:
        excess_return_cov_uncertainty = 1 / clean_period_prices.shape[0]

    reset_start_date = rqdatac.get_next_trading_date(reset_start_date)
    # Take daily risk free rate
    if risk_free_rate_tenor is None:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor='0S', country='cn')
    elif risk_free_rate_tenor in risk_free_rate_dict:
        risk_free_rate = rqdatac.get_yield_curve(reset_start_date, end_date, tenor=risk_free_rate_tenor,
                                                 country='cn')
    risk_free_rate['Daily'] = pd.Series(np.power(1 + risk_free_rate['0S'], 1 / 365) - 1, index=risk_free_rate.index)

    # Calculate daily risk premium for each equity
    clean_period_prices_pct_change = clean_period_prices.pct_change()
    clean_period_excess_return = clean_period_prices_pct_change.subtract(risk_free_rate['Daily'], axis=0)

    # Wash out the ones in kick_out_list
    clean_market_weight = market_weight.loc[clean_period_prices.columns.values]
    temp_sum_weight = clean_market_weight.sum()
    clean_market_weight = clean_market_weight.div(temp_sum_weight)

    # If no risk_aversion_coefficient is passed in, then
    # risk_aversion_coefficient = market portfolio risk premium / market portfolio volatility
    if risk_aversion_coefficient is None:
        market_portfolio_return = np.dot(clean_period_prices_pct_change, clean_market_weight)
        risk_aversion_coefficient = ((market_portfolio_return[1:].mean()-risk_free_rate["Daily"].mean()) /
                                     market_portfolio_return[1:].var())

    equilibrium_return = np.multiply(np.dot(clean_period_excess_return[1:].cov(), clean_market_weight),
                                     risk_aversion_coefficient)

    clean_period_excess_return_cov = clean_period_excess_return[1:].cov()
    # Generate the investors_views_uncertainty matrix if none is passed in
    if investors_views_uncertainty is None:
        if confidence_of_views is None:
            # He and Litteman's(1999) method to generate the uncertainty diagonal matrix, confidence level on each view
            # doesn't need.
            Omeg_diag = list()
            for i in range(investors_views_indicate_M.shape[0]):
                temp = np.dot(np.dot(investors_views_indicate_M[i, :], clean_period_excess_return_cov),
                              investors_views_indicate_M[i, :].transpose()) * excess_return_cov_uncertainty
                Omeg_diag.append(temp.item(0))
            investors_views_uncertainty = np.diag(Omeg_diag)
        else:
            # Idzorek's(2002) method, users can specify their confidence level on each view.
            Omeg_diag = list()
            for i in range(len(investors_views)):
                part1 = excess_return_cov_uncertainty * np.dot(clean_period_excess_return_cov,
                                                               investors_views_indicate_M[i, :].transpose())
                part2 = 1 / (excess_return_cov_uncertainty*np.dot(investors_views_indicate_M[i, :],
                                                                  np.dot(clean_period_excess_return_cov,
                                                                         investors_views_indicate_M[i, :].transpose())))
                part3 = investors_views[i]-np.dot(investors_views_indicate_M[i, :], equilibrium_return)
                return_with_full_confidence = equilibrium_return + np.multiply(part2 * part3, part1)
                weights_with_full_confidence = np.dot(np.linalg.inv(np.multiply(risk_aversion_coefficient,
                                                                    clean_period_excess_return_cov)),
                                                      return_with_full_confidence)
                temp1 = weights_with_full_confidence-clean_market_weight
                temp2 = np.multiply(confidence_of_views[i], np.absolute(investors_views_indicate_M[i, :].transpose()))
                tilt = np.multiply(temp1, temp2)
                weights_with_partial_confidence =clean_market_weight.as_matrix() + tilt

                def objective_fun(x):
                    temp1 = np.linalg.inv(np.multiply(risk_aversion_coefficient, clean_period_excess_return_cov))
                    temp2 = np.linalg.inv(np.linalg.inv(np.multiply(excess_return_cov_uncertainty,
                                                                    clean_period_excess_return_cov)) +
                                          np.multiply(np.reciprocal(x), np.dot(investors_views_indicate_M[i, :].transpose(),
                                                     investors_views_indicate_M[i, :])))
                    temp3 = (np.dot(np.linalg.inv(np.multiply(excess_return_cov_uncertainty,
                                                             clean_period_excess_return_cov)), equilibrium_return) +
                             np.multiply(investors_views[i]*np.reciprocal(x),
                                         investors_views_indicate_M[i, :].transpose()))
                    wk = np.dot(temp1, np.dot(temp2, temp3))
                    return np.linalg.norm(np.subtract(weights_with_partial_confidence, wk))

                # Upper bound should be consistent with the magnitude of return
                upper_bound = equilibrium_return.mean()*100
                omega_k = sc_opt.minimize_scalar(objective_fun, bounds=(10**-8, upper_bound), method="bounded",
                                                 options={"xatol": 10**-8})
                Omeg_diag.append(omega_k.x.item(0))
            investors_views_uncertainty = np.diag(Omeg_diag)

    # Combine all the information above to get the distribution of expected return with given views
    combined_return_covar = np.linalg.inv(np.linalg.inv(np.multiply(excess_return_cov_uncertainty, 
                                                                      clean_period_excess_return_cov))
                                            + np.dot(np.dot(investors_views_indicate_M.transpose(),
                                                            np.linalg.inv(investors_views_uncertainty)),
                                                     investors_views_indicate_M))
    temp1 = np.dot(np.linalg.inv(np.multiply(excess_return_cov_uncertainty, clean_period_excess_return_cov)), 
                   equilibrium_return)
    temp2 = np.dot(np.dot(investors_views_indicate_M.transpose(), np.linalg.inv(investors_views_uncertainty)),
                   investors_views)
    temp = temp1 + temp2

    combined_return_mean = np.dot(combined_return_covar, temp)
    return combined_return_mean, combined_return_covar, risk_aversion_coefficient, investors_views_uncertainty

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']

# N*1 vector
investors_views = np.matrix([[0.001], [0.002]])
# K*N matrix
investors_views_indicate_M = np.matrix([[0, -1, 0, 1, 0],
                                       [-1, 0, 1, 0, 0]])
# A list with K elements
confidence_of_views_list = [0.1, 0.1]

res = black_litterman_prep(equity_funds_list, "2015-01-01", investors_views, investors_views_indicate_M,
                           asset_type='fund', confidence_of_views=confidence_of_views_list)

print(res[0])
print(res[1])
print(res[2])
print(res[3])

