import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")

import datetime as dt
import itertools
import matplotlib.pyplot as plt
import numpy as np
from math import *
import pandas as pd
from optimizer_tests.fund_test.ptfopt import *



def get_fund_test_suite(before_date, big = 0):
    """
    Args:
    :param before_date: str
        generate test_suite before this date
    :param big: int
        an indicator to whether we want get big test suite
    :return fund_test_suite: dic
        a dictionary, e.x.  {'Bond_1 x 2=2': ['161216', '166003']
                                   the key is in this format for the convenience of saving plots
                                   "Bond_1 x 80=80" means Bond type, in total this combination involves 1 type,
                                   we select 80 out of 1 type and in total the length is 80 (this is calculated for proofread)
    """
    fund_pool = {}
    combo = []
    fund_test_suite = {}

    all_fund = fund.all_instruments(date=before_date)[['order_book_id', 'listed_date', 'symbol', 'fund_type']]
    # fund_types = np.unique(all_fund.fund_type)
    # currently we ruled out Other and Money type
    fund_types = ['Bond', 'BondIndex', 'Hybrid', 'QDII', 'Related', 'Stock', 'StockIndex']
    len_fund_types = len(fund_types)

    # get a dictionary of all_fund according to fund_types
    for i in fund_types:
        fund_pool[i] = (all_fund[all_fund['fund_type'] == i]['order_book_id'].values[:])

    # get all possible combinations of fund_types
    for j in range(1, len_fund_types + 1):
        for subset in itertools.combinations(fund_types, j):
            combo.append(subset)

    # We have tried all the following cases:
    # each_fund_num = {1: [5, 6, 7, 8], 2: [3,4,5,6], 3: [2,3,4], 4:[1,2,3], 5:[1,2], 6:[1,2], 7:[1]}
    # , 8:[1], 9:[1]}
    # each_fund_num = {1: [ 6, 7, 8], 2: [3,4,5,6], 3: [2,3,4], 4:[2,3], 5:[2], 6:[1,2], 7:[1]}
    # each_fund_num = {1: [7, 8], 2: [4, 5, 6], 3: [3, 4], 4: [2, 3], 5: [2], 6: [2], 7: [1]}

    # each_fund_num here is to specify how many we want out of one type
    if big == 0:
        each_fund_num = {1: [8], 2: [4, 6], 3: [3], 4: [2], 5: [2], 6: [2], 7: [1]}
    else:
        each_fund_num = {1: [80,100], 2: [50], 3: [],4: [], 5: [], 6: [], 7: []}

    for k in combo:
        len_combo = len(k) # the length of combo indicates how many types we selected
        for l in each_fund_num[len_combo]:
            temp = [a for x in k for a in fund_pool[x][0:l]]
            fund_test_suite['_'.join(k)+'_'+str(len_combo)+' x '+str(l)+'='+str(len(temp))] = temp

    return fund_test_suite




######### example:
# fund_test_suite = get_fund_test_suite('2014-01-01')
# len(fund_test_suite)
# 379


# we also tried the following total number test_suite
# 246
# 148



# fund_test_suite_large = get_fund_test_suite('2014-01-01', 1)
# len(fund_test_suite_large)
# 35



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
        methods = ['risk_parity', 'min_variance', "risk_parity_with_con"]
    else:
        methods = method


    # call ptfopt2.py and run 'optimizer'
    opt_res = {}
    weights = {}
    c_m = {}
    indicators = {}
    key_a_r = methods+['equal_weight']
    daily_methods_period_price = {x: pd.Series() for x in key_a_r}
    daily_methods_a_r = {x: pd.Series() for x in key_a_r}
    for i in range(0, count + 1):
        # optimizer would return:
        # if all kicked out: kicked out list
        # if not all kicked out: weight, cov_mtrx, kicked out list
        if bc == 0:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method, fun_tol=10**-8)
        elif bc == 1:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   bnds = {'full_list': (0, 0.2)}, fun_tol=10**-12)
        elif bc == 2:
            # opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
            #                        cons = {name[:name.find('_')]: (0.6, 1)} )
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   cons= 1, fun_tol=10**-8)
        elif bc == 3:
            opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   bnds={'full_list': (0, 0.2)}, cons = 1, fun_tol=10**-8)

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
            period_data = rqdatac.get_price(assets_list, time_frame[i], time_frame[i+1], frequency='1d', fields=['close'])
            period_prices = period_data['close']


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

    plt.savefig('./optimizer_tests/fund_test/result/normal/figure/test_res' + str(bc) + '/%s' % (name))
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




def get_efficient_plots(fund_test_suite, bigboss):
    """
    To generate the plot of all test results, x-axis is annualized_vol, y-axis is annualized return
    :param fund_test_suite:
    :param bigboss (dic): a dictionary stored all test results
    :return:
    """
    annualized_vol = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "risk_parity_with_con": []}
    annualized_return = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "risk_parity_with_con": []}


    for j in ['equal_weight', 'min_variance', 'risk_parity', "risk_parity_with_con"]:
        for i in fund_test_suite.keys():
            annualized_vol[j].append(bigboss[i][2][j])
            annualized_return[j].append(bigboss[i][1][j])

        plt.figure(1, figsize=(10, 8))
        p1 = plt.plot(annualized_vol[j], annualized_return[j], 'o', label = j)
    plt.legend()
    plt.xlabel('annualized_vol')
    plt.ylabel('annualized_return')
        #plt.show()
    plt.savefig('./optimizer_tests/fund_test/result/normal/figure/%s' % (name))
    plt.close()

    return p1



# test_fund_opt is to wrap get_optimizer to run all suite
def test_fund_opt(fund_test_suite, bc = 0):
    """
    :param fund_test_suite (dic)
    :param bc (int): an indicator about bounds and constraints. 0 means nothing;
                                                               1 means have bounds, no constraints;
                                                               2 means no bounds, have constraints;
                                                               3 means have both
    :return:
    """
    a = {}
    for k in fund_test_suite.keys():
        a[k] = get_optimizer(fund_test_suite[k],  start_date = '2014-01-01',end_date= '2017-05-31',
                             asset_type='fund', method='all', fields ='all', name = k,bc = bc)

    return a

