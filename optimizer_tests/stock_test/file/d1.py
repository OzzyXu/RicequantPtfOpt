# 导入相关的 Python 库

import numpy as np
import statsmodels as sm
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import statsmodels.api as sm

import rqdatac
# 填写 RQData 账号和密码

from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")

SHENWAN_INDUSTRY_MAP = {
    "801010.INDX": "农林牧渔",
    "801020.INDX": "采掘",
    "801030.INDX": "化工",
    "801040.INDX": "钢铁",
    "801050.INDX": "有色金属",
    "801080.INDX": "电子",
    "801110.INDX": "家用电器",
    "801120.INDX": "食品饮料",
    "801130.INDX": "纺织服装",
    "801140.INDX": "轻工制造",
    "801150.INDX": "医药生物",
    "801160.INDX": "公用事业",
    "801170.INDX": "交通运输",
    "801180.INDX": "房地产",
    "801200.INDX": "商业贸易",
    "801210.INDX": "休闲服务",
    "801230.INDX": "综合",
    "801710.INDX": "建筑材料",
    "801720.INDX": "建筑装饰",
    "801730.INDX": "电气设备",
    "801740.INDX": "国防军工",
    "801750.INDX": "计算机",
    "801760.INDX": "传媒",
    "801770.INDX": "通信",
    "801780.INDX": "银行",
    "801790.INDX": "非银金融",
    "801880.INDX": "汽车",
    "801890.INDX": "机械设备"
}

benchmark_code = "000300.XSHG"  # 沪深300
date = '20161229'


i_c = rqdatac.index_components(benchmark_code, date)
benchmark = pd.DataFrame(index=i_c)
benchmark["industry"] = [rqdatac.shenwan_instrument_industry(s, date) for s in benchmark.index]
market_cap = rqdatac.get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap).filter(fundamentals.eod_derivative_indicator.stockcode.in_(benchmark.index)), entry_date=date)




market_cap_ = market_cap.loc[market_cap.items[0]].transpose()
result = pd.concat([benchmark, market_cap_], axis=1)
column_names = result.columns.values
column_names[1] = "market_cap"
result.columns = column_names
result


result[result.industry==("801790.INDX", "非银金融")].sort_values("market_cap",ascending=False)




def get_stock_test_suite(start_t = '2013-01-01', end_t = '2017-07-05'):

    # get all stocks
    all_stocks0 = list(all_instruments(type = 'CS').order_book_id)

    # make sure stocks are alive during start_t ~ end_t
    all_stocks1= [i.order_book_id for i in instruments(all_stocks0) if i.listed_date <= start_t and
                  (i.de_listed_date == '0000-00-00' or end_t < i.de_listed_date)]

    # rule out ST stocks
    temp0 = is_st_stock(all_stocks1, start_t, end_t).sum(axis = 0)
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
    temp2["industry"] = [shenwan_instrument_industry(s) for s in temp2.index] # don't add date to shenwan_instrument_industry
    shenwan_name = temp2.industry.unique()


    # get stock pool

    stock_test_suite = {}
    stock_pool = {}

    # temp2 is sorted by market cap
    stock_test_suite[0] = list(temp2.index[:100])
    stock_test_suite[1] = list(temp2.index[100:200])
    stock_test_suite[2] = list(temp2.index[-200:-100])
    stock_test_suite[3] = list(temp2.index[-100:])
    stock_test_suite[4] = list(temp2.index[:50]) + list(temp2.index[-50:])

    # temp3 is sorted by industry first and then within industry by market cap in descending order
    temp3 = temp2.sort_values(by = ['industry', 'market_cap'], ascending = False)

    # within industry tag them with [1,2,3] to split them into 3 categories
    for i in shenwan_name:
        index0 = temp3['industry'] == i
        len0 = sum(index0)
        len0_int = int(len0/3)
        len0_residual = len0 % 3
        cate_temp = list(np.repeat([1,2,3], len0_int)) + [3]*len0_residual
        temp3.loc[index0,'category'] = cate_temp

    # get the number of stocks within each industry
    sum_info = temp3.groupby(by='industry').size()
    safe_num = min(sum_info)/3


    for i in range(5,100):
        stock_test_suite[i] = [temp3.loc[temp3.industry == a].loc[temp3.category == b].index[np.random.randint(safe_num)]
                               for a in shenwan_name for b in [1,2,3] ]

   #s =  [temp3.loc[temp3.industry == a].loc[temp3.category == b].index   for a in shenwan_name for b in [1,2,3] ]

    return stock_test_suite








a = get_stock_test_suite()



temp3.groupby(by = 'industry').agg('count')


    # create a data frame dictionary to store your data frames
    stock_pool = {elem: pd.DataFrame for elem in key0}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = data[:][data.Names == key]




    data = pd.DataFrame(
        {'Names': ['Joe', 'John', 'Jasper', 'Jez'] * 4, 'Ob1': np.random.rand(16), 'Ob2': np.random.rand(16)})

    # create unique list of names
    UniqueNames = data.Names.unique()





    df = data

    df.sort(columns=['name'], inplace=True)
    # set the index to be this and don't drop
    df.set_index(keys=['name'], drop=False, inplace=True)
    # get a list of names
    names = df['name'].unique().tolist()
    # now we can perform a lookup on a 'view' of the dataframe
    joe = df.loc[df.name == 'Joe']
    # now you can query all 'joes'




    shenwan_instrument_industry('600649.XSHG')
    instruments('600649.XSHG')

    stock_pool
    {i} = temp1



    stock_pool = {}
    combo = []
    stock_test_suite = {}

a_year_before = date - datetime.timedelta(days=365)

a_year_before_str = str(a_year_before)
date_str = str(date)

listed_stocks = [i for i in instruments(list(all_instruments(type='CS').order_book_id)) if
                 i.listed_date <= start_t and
                 (i.de_listed_date == '0000-00-00' or end_t < i.de_listed_date)]








set(a2['type'])

len(set(a2['type']))

3357


def get_stock_test_suite(before_date, big = 0):
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
    stock_pool = {}
    combo = []
    stock_test_suite = {}

    for i in SHENWAN_INDUSTRY_MAP.values():
        stock_list =  shenwan_industry(i)

        market_cap = rqdatac.get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap).filter(
            fundamentals.eod_derivative_indicator.stockcode.in_(stock_list)), entry_date=date)
        market_cap_ = market_cap.loc[market_cap.items[0]].transpose()

        stock_df = pd.DataFrame(index = stock_list)
        temp = pd.concat([stock_df, market_cap_], axis=1)
        temp.columns = ['market_cap']
        temp1 = temp.sort_values(by = 'market_cap', ascending = 0)  # descending sort by market value
        stock_pool{i} = temp1









        market_values = get_market_value(stock_list, start_date, end_date)
        ncol = market_values.shape[1]
        if ncol < threshold_num:
            stock_test_suite[i] = stock_list
        else:
            df1 = market_values.iloc[:, np.argsort(market_values.loc[end_date])]
            num1 = int(threshold_num / 3)
            stocks_picked = list(df1.columns[: num1]) + \
                            list(df1.columns[num1 + 1: num1 + num1]) + \
                            list(df1.columns[-num1:])
            # list(range(1, threshold_num/3)), listthreshold_num/3+1:]
            stock_test_suite[i] = stocks_picked


    all_stock = fund.all_instruments(date=before_date)[['order_book_id', 'listed_date', 'symbol', 'fund_type']]
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




    for i in shenwan_industry_names:
        stock_list = shenwan_industry(i)
        market_values = get_market_value(stock_list, start_date, end_date)
        ncol = market_values.shape[1]
        if  ncol < threshold_num:
            stock_test_suite[i] = stock_list
        else:
            df1 = market_values.iloc[:, np.argsort(market_values.loc[end_date])]
            num1 = int(threshold_num/3)
            stocks_picked = list(df1.columns[: num1]) +  \
                            list(df1.columns[num1 + 1 : num1 + num1])+ \
                            list(df1.columns[-num1:])
                            #list(range(1, threshold_num/3)), listthreshold_num/3+1:]
            stock_test_suite[i] = stocks_picked

    return(stock_test_suite)