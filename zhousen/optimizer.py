##############################
# created by zs on June 29th
##############################



import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")


import datetime as dt
import itertools


import matplotlib.pyplot as plt



# wrap ptfopt.py to perform as some general API


def get_optimizer(order_book_ids, start_date, asset_type, method, tr_frequency = 66, current_weight=None, bnds=None,
                  cons=None, expected_return=None, expected_return_covar=None, risk_aversion_coefficient=1, fields = 'weights',
                  end_date = dt.date.today().strftime("%Y-%m-%d"), name = None, bc = 0):
    """
    :param order_book_ids:
    :param start_date:
    :param asset_type:
    :param method:
    :param tr_frequency:
    :param current_weight:
    :param bnds:
    :param cons:
    :param expected_return:
    :param expected_return_covar:
    :param risk_aversion_coefficient:
    :param fields:
    :param end_date:
    :param name:
    :param bc:
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



    # call ptfopt.py and run 'optimizer'
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
                                   bnds = {'full_list': (0, 0.2)}, fun_tol=10**-8)
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

    plt.savefig('./figure/test_res'+str(bc)+'/%s' % (name))
    plt.close()


    return_pack = {'weights': weights, 'annualized_return': annualized_return,
                   'annualized_vol': annualized_vol, 'indicators': indicators, 'mmd': mmd}
    if fields == 'all':
        #fields = ['weights', 'annualized_return','annualized_vol','indicators','mmd']
        fields = ['weights', 'annualized_return', 'annualized_vol']

    return [return_pack[x] for x in fields]




def test_fund_opt(fund_test_suite, bc = 0):
    a = {}
    for k in fund_test_suite.keys():
        a[k] = get_optimizer(fund_test_suite[k],  start_date = '2014-01-01',end_date= '2017-05-31',
                             asset_type='fund', method='all', fields ='all', name = k,bc = bc)

    return a

################
plt.switch_backend('MacOSX')

bigboss = test_fund_opt(fund_test_suite)


bigboss_b_nc = test_fund_opt(fund_test_suite, 1)


bigboss_nb_c = test_fund_opt(fund_test_suite, 2)

bigboss_b_c = test_fund_opt(fund_test_suite, 3)


## test for large

bigboss = test_fund_opt(fund_test_suite_large)


bigboss_b_nc = test_fund_opt(fund_test_suite_large, 1)


bigboss_nb_c = test_fund_opt(fund_test_suite_large, 2)

bigboss_b_c = test_fund_opt(fund_test_suite_large, 3)







###########

def get_optimizer_indicators(weight0, cov_matrix, asset_type, type_tag=1):
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



a = pd.DataFrame(index = x0.index, columns =['z'] )
a.shape
a.iloc[:,0] = [1,2,3,8,3,5,8,3,4,5]

x = a

x = x0
list(x0.values)

def get_maxdrawdown(x):
    # x need to pandas type
    x_cummax = x.cummax()
    cmaxx = x_cummax - x
    cmaxx_p = cmaxx/x_cummax
    mdd = cmaxx_p.max()
    pos1 = [i for i in cmaxx_p.index if cmaxx_p.loc[i]== mdd]

    pos0 =[]
    time0 = [cmaxx_p.index[0]] + pos1
    for i in range(len(pos1)):
        data0 = cmaxx_p.loc[time0[0]:time0[i+1]]
        t = max(data0[(data0 == 0)].index)
        pos0.append(t)

    return mdd, pos0, pos1


get_maxdrawdown(x0)


x = pd.Series([1,2,5,3,2,8,5,4,6])
x.plot()



qq1 = weights[i][j]
type(qq1)
weight0 = qq1
cov_matrix = c_m[i]
get_optimizer_indicators(qq1, c_m[i], 'fund')

(0.083336409037438566, 0, 0.083333333366530093, nan, 0, nan)







def wrap_and_run(test_suite, start_date, end_date, adjust_frequency = 66):

    test_res = pd.DataFrame(columns=['risk_parity_return', 'min_variance_return', 'equal_weight_return',
                                             'risk_parity_sigma', 'min_variance_sigma', 'equal_weight_sigma'])
    for i in test_suite.keys():
        df1 = pd.DataFrame(index=[i], columns=['risk_parity_return', 'min_variance_return', 'equal_weight_return',
                                             'risk_parity_sigma', 'min_variance_sigma', 'equal_weight_sigma'])
        equity_funds_list = test_suite[i]
        # try:
        #     a = optimizer_test(equity_funds_list, start_date, end_date, adjust_frequency, name=i)
        # except:
        #     print('outside:', i)
        a = optimizer_test(equity_funds_list, start_date, 'fund', end_date = end_date, method = 'all', adjust_frequency = adjust_frequency, name=i)

        if a is None:
            pass
        else:
            df1['risk_parity_return'] = a[1]['risk_parity']
            df1['risk_parity_sigma'] = a[2]['risk_parity']

            df1['min_variance_return'] = a[1]['min_variance']
            df1['min_variance_sigma'] = a[2]['min_variance']

            df1['equal_weight_return'] = a[1]['equal_weight']
            df1['equal_weight_sigma'] = a[2]['equal_weight']
            test_res = pd.concat([test_res, df1])


    return test_res



import json
with open('./fund_test_suite_0704.txt', 'w') as file:
    file.write(json.dumps(fund_test_suite, ensure_ascii=False))



with open('./bigboss_0705.txt', 'w') as file:
    file.write(json.dumps(bigboss, ensure_ascii=False))

json.dump(bigboss, open("text.txt",'w'))



import pickle

pickle.dump(bigboss, open( "bigboss_0705.p", "wb" ) )

pickle.dump(bigboss_b_nc, open( "bigboss_b_nc_0705.p", "wb" ) )

pickle.dump(bigboss_nb_c, open( "bigboss_nb_c_0705.p", "wb" ) )

pickle.dump(bigboss_b_c, open( "bigboss_b_c_0705.p", "wb" ) )



qq1 = pickle.load(open( "bigboss_0705.p", "rb" ))

# ## 如需重新读入使用：
# with open('./common/stock_test_suite.txt', 'r') as f:
#     d1 = json.load(f)




zs0 = get_optimizer(order_book_ids, start_date, asset_type = 'fund', method = 'all')


order_book_ids = fund_test_suite['Related_Stock_StockIndex_3 x 4=12']


len(fund_test_suite)




