import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")

import datetime as dt
import itertools
import matplotlib.pyplot as plt
import numpy as np
from math import *
import pandas as pd
from optimizer_tests.stock_test.ptfopt import *



def get_stock_test_suite(start_t='2013-01-01', end_t='2017-07-05'):
    """
    get alive stock test suite between dates (for test use between 20140101 to 2017)
    make sure it has IPO for at least one year and is never ST between dates
    :param start_t:
    :param end_t:
    :return: dic
        return a dic, key is 0-99.
        0 is the biggest 100
        1 is the second 101 ~ 200 stocks
        2 is the smallest -200 ~ -101 stocks
        3 is the smallest -100 ~ -1 stocks
        4 is the biggest 50 + smallest 50
        5 ~ 99 is the 28*3 combo (28: shenwan_industry category, 3: we split each cate by market cap)
    """
    # get all stocks
    all_stocks0 = list(all_instruments(type='CS').order_book_id)

    # make sure stocks are alive during start_t ~ end_t
    all_stocks1 = [i.order_book_id for i in instruments(all_stocks0) if i.listed_date <= start_t and
                   (i.de_listed_date == '0000-00-00' or end_t < i.de_listed_date)]

    # rule out ST stocks
    temp0 = is_st_stock(all_stocks1, start_t, end_t).sum(axis=0)
    all_stocks2 = [i for i in all_stocks1 if temp0.loc[i] == 0]

    # calculate all their market_cap
    market_cap = rqdatac.get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap).filter(
        fundamentals.eod_derivative_indicator.stockcode.in_(all_stocks2)), entry_date='20140101')
    market_cap_ = market_cap.loc[market_cap.items[0]].transpose()
    stock_df = pd.DataFrame(index=all_stocks2)
    temp1 = pd.concat([stock_df, market_cap_], axis=1)
    temp1.columns = ['market_cap']
    temp2 = temp1.sort_values(by='market_cap', ascending=False)  # descending sort by market value

    # tag them with shenwan category
    temp2["industry"] = [shenwan_instrument_industry(s) for s in
                         temp2.index]  # don't add date to shenwan_instrument_industry
    shenwan_name = temp2.industry.unique()

    stock_test_suite = {}

    # notice that temp2 is sorted by market cap
    stock_test_suite[0] = list(temp2.index[:100])
    stock_test_suite[1] = list(temp2.index[100:200])
    stock_test_suite[2] = list(temp2.index[-200:-100])
    stock_test_suite[3] = list(temp2.index[-100:])
    stock_test_suite[4] = list(temp2.index[:50]) + list(temp2.index[-50:])

    # temp3 is sorted by industry first and then within industry by market cap in descending order
    temp3 = temp2.sort_values(by=['industry', 'market_cap'], ascending=False)

    # within industry tag them with [1,2,3] to split them into 3 categories
    for i in shenwan_name:
        index0 = temp3['industry'] == i
        len0 = sum(index0)
        len0_int = int(len0 / 3)
        len0_residual = len0 % 3
        cate_temp = list(np.repeat([1, 2, 3], len0_int)) + [3] * len0_residual
        temp3.loc[index0, 'category'] = cate_temp

    # get the number of stocks within each industry
    sum_info = temp3.groupby(by='industry').size()
    safe_num = min(sum_info) / 3  # this number is for randint() use

    for i in range(5, 100):
        stock_test_suite[i] = [
            temp3.loc[temp3.industry == a].loc[temp3.category == b].index[np.random.randint(safe_num)]
            for a in shenwan_name for b in [1, 2, 3]]


    return stock_test_suite




######### example:
# stock_test_suite = get_stock_test_suite()
# len(stock_test_suite)
# 100



# Save things as pickle file
# import pickle
# pickle.dump(stock_test_suite, open( "./optimizer_tests/stock_test/file/stock_test_suite.p", "wb" ))
# To read it back to use again
# temp = pickle.load(open( "./optimizer_tests/stock_test/file/stock_test_suite.p", "rb" ))








def get_optimizer(order_book_ids, start_date, asset_type, method, tr_frequency = 66, current_weight=None, bnds=None,
                  cons=None, expected_return=None, expected_return_covar=None, risk_aversion_coefficient=1, fields = 'weights',
                  end_date = '20170705', name = None, bc = 0):
    """
    Wrap ptfopt2.py to run optimizer and get indicators
    :param order_book_ids: list. A list of assets(stocks or funds);
    :param start_date: str. Date to initialize a portfolio or rebalance a portfolio;
    :param asset_type: str or str list. Types of portfolio candidates,  "stock" or "fund", portfolio with mixed assets
                       is not supported;
    :param method: str. Portfolio optimization model: "risk_parity", "min_variance", "mean_variance",
                        "risk_parity_with_con", "all"("all" method only contains "risk_parity", "min_variance",
                        "risk_parity_with_con" but not "mean_variance");
    :param tr_frequency: int. number of days to rebalance
    :param current_weight:
    :param bnds: list of floats. Lower bounds and upper bounds for each asset in portfolio.
                Support input format: {"asset_code1": (lb1, up1), "asset_code2": (lb2, up2), ...}
                or {'full_list': (lb, up)} (set up universal bounds for all assets);
    :param cons: dict, optional. Lower bounds and upper bounds for each category of assets in portfolio;
                        Supported funds type: Bond, Stock, Hybrid, Money, ShortBond, StockIndex, BondIndex,
                         Related, QDII, Other; supported
                        stocks industry sector: Shenwan_industry_name;
                        cons: {"types1": (lb1, up1), "types2": (lb2, up2), ...};
    :param expected_return: column vector of floats, optional. Default: Means of the returns of order_book_ids
                            within windows.
    :param expected_return_covar: numpy matrix, optional. Covariance matrix of expected return. Default: covariance of
                            the means of the returns of order_book_ids within windows;
    :param risk_aversion_coefficient: float, optional. Risk aversion coefficient of Mean-Variance model. Default: 1.
    :param fields: str. specify what to return
    :param end_date: str. specify test ending time
    :param name: str. For naming the plots, using the fund_test_suite keys
    :param bc: int. For specify whether we have bnds and cons
    :return:
    """


    # according to start_date and tr_frequency to determine time points
    trading_date_s = get_previous_trading_date(start_date)
    trading_date_e = get_previous_trading_date(end_date)
    trading_dates = get_trading_dates(trading_date_s, trading_date_e)
    time_len = len(trading_dates)
    count = floor(time_len / tr_frequency)

    time_frame = {}
    for i in range(0, count+1):
        time_frame[i] = trading_dates[i * tr_frequency]
    time_frame[count+1] = trading_dates[-1]

    # determine methods name for later use
    if method == 'all':
        methods = ['risk_parity', 'min_variance', 'mean_variance']
    else:
        methods = method


    # call ptfopt2.py and run 'optimizer'
    opt_res = {}
    weights = {}
    c_m = {}
    indicators = {}
    key_a_r = list(methods) +['equal_weight']
    daily_methods_period_price = {x: pd.Series() for x in key_a_r}
    daily_methods_a_r = {x: pd.Series() for x in key_a_r}
    for i in range(0, count + 1):
        # optimizer would return:
        # if all kicked out: kicked out list
        # if not all kicked out: weight, cov_mtrx, kicked out list
        if bc == 0:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method, fun_tol=10**-6)
        elif bc == 1:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   bnds = {'full_list': (0, 0.015)}, fun_tol=10**-6)
        elif bc == 2:
            # opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
            #                        cons = {name[:name.find('_')]: (0.6, 1)} )
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   cons= 1, fun_tol=10**-6)
        elif bc == 3:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   bnds={'full_list': (0, 0.025)}, cons = 1, fun_tol=10**-6, iprint = 2, disp = True)

        # if all assets have been ruled out, print and return -1
        if len(opt_res[i]) == 1:
            print('All selected ' + asset + ' have been ruled out')
            #return -1

        weights[i] = opt_res[i][0]
        c_m[i] = opt_res[i][1]
        assets_list = [x for x in weights[i].index]

        if asset_type is 'fund':
            period_prices = fund.get_nav(assets_list, time_frame[i], time_frame[i+1], fields='adjusted_net_value')
        elif asset_type is 'stock':
            period_prices = rqdatac.get_price(assets_list, time_frame[i], time_frame[i+1], frequency='1d', fields=['close'])
            #period_prices = period_data['close']


        period_daily_return_pct_change = period_prices.pct_change()

        if i != 0:
            period_daily_return_pct_change = period_daily_return_pct_change[1:]
            period_prices = period_prices[1:]


        # calculate portfolio arithmetic return by different methods
        equal_weight0 = [1 / len(assets_list)] * len(assets_list)
        weights[i]['equal_weight'] = equal_weight0
        # weighted_sum0 = period_daily_return_pct_change.multiply(equal_weight0).sum(axis=1)
        # daily_methods_a_r['equal_weight'] = daily_methods_a_r['equal_weight'].append(weighted_sum0)
        for j in key_a_r:
            # calculate sum of daily return pct_change
            weighted_sum1 = period_daily_return_pct_change.multiply(weights[i][j]).sum(axis=1)
            daily_methods_a_r[j] = daily_methods_a_r[j].append(weighted_sum1)

            #indicators[j] = get_optimizer_indicators(weights[i][j], c_m[i], asset_type=asset_type)

            # calculate sum of daily price
            #weighted_sum2 = period_prices.multiply(weights[i][j]).sum(axis=1)
            #daily_methods_period_price[j] = daily_methods_period_price[j].append(weighted_sum2)



    annualized_vol = {}
    annualized_return = {}
    mmd = {}
    for j in (key_a_r):

        #s0 = pd.Series([0], index = [dt.datetime.strftime(time_frame[0], '%Y-%m-%d %H:%M:%S')])
        #daily_methods_a_r[j] = pd.Series.append(s0, daily_methods_a_r[j])
        daily_methods_a_r[j][0] = 0
        temp = np.log(daily_methods_a_r[j] + 1)
        annualized_vol[j] = sqrt(244) * temp.std()
        days_count = len(daily_methods_a_r[j])
        daily_cum_log_return = temp.cumsum()
        annualized_return[j] = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1

        #mmd[j] = get_maxdrawdown(daily_methods_period_price[j])

        str1 = "%s: r = %f, $\sigma$ = %f, s_r = %f" % (j, annualized_return[j], annualized_vol[j],
                                                        annualized_return[j] / annualized_vol[j])


        plt.figure(1, figsize=(10, 8))
        p1 = daily_cum_log_return.plot(legend=True, label=str1)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3))

    plt.savefig('./optimizer_tests/stock_test/result/normal/figure/test_res'+str(bc)+'/%s' % (name))
    plt.close()


    return_pack = {'weights': weights, 'annualized_return': annualized_return,
                   'annualized_vol': annualized_vol, 'indicators': indicators, 'mmd': mmd}
    if fields == 'all':
        #fields = ['weights', 'annualized_return','annualized_vol','indicators','mmd']
        fields = ['weights', 'annualized_return', 'annualized_vol']

    return [return_pack[x] for x in fields]






def get_optimizer_indicators(weight0, cov_matrix, asset_type, type_tag=1):
    """
    To calculate highest rick contributor individually or grouply
    :param weight0: (list). weight at current time t
    :param cov_matrix: (np.matrix). cov_matrix calculated at current time t
    :param asset_type: (str). 'fund' or 'stock'
    :param type_tag: (int). indicator about whether we need group risk contributor or not. 0 means no, 1 means yes
    :return:
    """
    weight = np.array(weight0)
    # refer to paper formula 2.19 2.20
    production_i = weight * (cov_matrix.dot(weight))
    productions = weight.dot(cov_matrix).dot(weight)

    ## calculate for individual:
    # calculate hightest risk contributions
    HRCs = production_i / productions
    # index, value = max(enumerate(HRCs), key=operator.itemgetter(1))
    HRC, HRC_index = max([(v, i) for i, v in enumerate(HRCs)])

    # calculate Herfindahl
    Herfindahl = np.sum(HRCs ** 2)


    ## calculate for groups:
    # calculate hightest risk contributions

    df1 = pd.DataFrame(columns=['HRCs', 'type'])
    df1['HRCs'] = HRCs
    if asset_type is 'fund':
        for i in weight0.index:
            df1.loc[i, 'type'] = fund.instruments(i).fund_type
    elif asset_type is 'stock':
        for i in weight0.index:
            df1.loc[i, "type"] = rqdatac.instruments(i).shenwan_industry_name

    productionG_i = df1.groupby(['type'])['HRCs'].sum()
    HRCGs = productionG_i
    HRCG, HRCG_index = max([(v, i) for i, v in enumerate(HRCGs)])

    # calculate Herfindahl
    Herfindahl_G = np.sum(HRCGs ** 2)

    ## get asset code type
    HRC_id = HRCs.index[HRC_index]
    df1.loc[HRC_id]

    if type_tag == 0:
        return (HRC, HRCs.index(HRC_index), HRC_id, Herfindahl)
    else:
        return (HRC, HRCs.index[HRC_index], HRC_id, Herfindahl, HRCG, HRCGs.index[HRCG_index], Herfindahl_G)




def get_efficient_plots(fund_test_suite, bigboss, name):
    """
    To generate the plot of all test results, x-axis is annualized_vol, y-axis is annualized return
    :param fund_test_suite:
    :param bigboss (dic): a dictionary stored all test results
    :return:
    """
    annualized_vol = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "mean_variance": []}
    annualized_return = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "mean_variance": []}


    for j in ['equal_weight', 'min_variance', 'risk_parity', "mean_variance"]:
        for i in fund_test_suite.keys():
            annualized_vol[j].append(bigboss[i][2][j])
            annualized_return[j].append(bigboss[i][1][j])

        plt.figure(1, figsize=(10, 8))
        p1 = plt.plot(annualized_vol[j], annualized_return[j], 'o', label = j)
    plt.legend()
    plt.xlabel('annualized_vol')
    plt.ylabel('annualized_return')
        #plt.show()
    plt.savefig('./optimizer_tests/stock_test/result/normal/figure/%s' %(name))
    plt.close()

    return p1



# test_stock_opt is to wrap get_optimizer to run all suite
def test_stock_opt(stock_test_suite, bc = 0):
    """
    :param stock_test_suite (dic)
    :param bc (int): an indicator about bounds and constraints. 0 means nothing;
                                                               1 means have bounds, no constraints;
                                                               2 means no bounds, have constraints;
                                                               3 means have both
    :return:
    """
    a = {}
    for k in stock_test_suite.keys():
        a[k] = get_optimizer(stock_test_suite[k],  start_date = '2014-01-01',end_date= '2017-05-31',
                             asset_type='stock', method='all', fields ='all', name = k, bc = bc)

    return a