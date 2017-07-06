def get_optimizer(order_book_ids, start_date, asset_type, method, tr_frequency = 66, current_weight=None, bnds=None,
                  cons=None, expected_return=None, expected_return_covar=None, risk_aversion_coefficient=1, fields = 'weights',
                  end_date = dt.date.today().strftime("%Y-%m-%d"), name = None, bc = 0):
    """
    Wrap ptfopt.py to run optimizer and get indicators
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

