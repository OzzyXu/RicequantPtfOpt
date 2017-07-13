

import pandas as pd
import numpy as np
import json


import rqdatac
from rqdatac import *
rqdatac.init('ricequant','8ricequant8')






#####  统计每个行业对应的股票个数，最后返回一个shenwan_df的data frame

# read in 申万行业分类code和名称
def get_shenwan_info():
    with open('./optimizer_tests/stock_test/file/shenwan.txt', 'r') as f:
        data1 = f.readlines()
    shenwan_df = pd.DataFrame(index = range(1,29),columns = ['code', 'name', 'num'])
    for i in range(len(data1)):
        shenwan_df.iloc[i,0:2] = data1[i].rstrip('\n').split('\t')
    # 申万行业分类中文名称  shenwan_industry_names = shenwan_df['name']
    # 统计每个行业的股票数量
    for i in range(1,29):
        temp_all = shenwan_industry(shenwan_df.loc[i]['name'])
        shenwan_df.loc[i]['num'] = len(temp_all)

    # save as txt for later use
    #shenwan_df.to_csv(r'./optimizer_tests/stock_test/file/shenwan_df.txt', header=True, index=None, sep=' ', mode='a')
    return(shenwan_df)


# res = get_shenwan_info()
## 股票总个数
# sum(res['num'])
# 3270




# 我们暂时选取60只股票

def get_stock_test_suite(shenwan_industry_names, start_date='2017-05-25', end_date='2017-05-26', threshold_num = 60):
    stock_test_suite = {}
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




## 获取股票test suite，返回一个dictionary
stock_test_suite = get_stock_test_suite(shenwan_industry_names)



## 将返回的dictionary存成txt file

with open('./common/stock_test_suite.txt', 'w') as file:
    file.write(json.dumps(stock_test_suite, ensure_ascii=False))



## 如需重新读入使用：
with open('./common/stock_test_suite.txt', 'r') as f:
    d1 = json.load(f)






################################################################################

a1=  [a for a in result.index]
a1

get_market_value(a1, start_date='20161230', end_date='20161230')

for i in a1:
    result.loc[i]['zs'] = get_market_value(i, start_date = '20161230', end_date = '20161230')[0]




result['zs'] = 'NaN'

result.shape

result.head







################################################################################
#-------------------------------------------------------------------------------
#

def get_market_value(stock_list, start_date = '2017-06-1', end_date = '2017-6-1'):
    """
    以下函数用来求股票的市值  （市值 = 总股本 x 选定日期收盘价）
    Get market value of a stock or stock_list from 'start_date' to 'end_date'
    :param stock_list: str or str_list
        one stock id or a list of stock ids
    :param start_date: date
    :param end_date: date
    :return:
        if stock_list is str:
            output a series of market value
        if stock_list is str_list:
            output pandas data frame
    """
    p1 = get_price(stock_list, start_date = start_date, end_date = end_date, fields = 'close')
    s1 = get_shares(stock_list, start_date, end_date, fields = 'total')
    if isinstance(stock_list, str):
        return(p1*s1)
    else:
        market_value = pd.DataFrame(index = p1.index, columns = p1.columns)
        for i in p1.columns:
            market_value[i] = p1[i] * s1[i]

        res = market_value
        return(res)



# ex:
# stock_list =[ '601998.XSHG','603323.XSHG']
# res = get_market_value(stock_list)

# stock_list = '601998.XSHG'
# get_market_value(stock_list)



################################################################################
# -------------------------------------------------------------------------------
#
