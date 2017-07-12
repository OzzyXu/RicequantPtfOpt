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





