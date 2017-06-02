import pandas as pd
import numpy as np
import json


import rqdatac
from rqdatac import *
rqdatac.init('ricequant','8ricequant8')



#####  统计每个行业对应的股票个数，最后返回一个shenwan_df的data frame

# read in 申万行业分类code和名称
with open('./common/shenwan.txt', 'r') as f:
    data1 = f.readlines()
#type(data1)


shenwan_df = pd.DataFrame(index = range(1,29),columns = ['code', 'name', 'num'])
for i in range(len(data1)):
    shenwan_df.iloc[i,0:2] = data1[i].rstrip('\n').split('\t')

# 申万行业分类中文名称
shenwan_industry_names = shenwan_df['name']

# 统计每个行业的股票数量
num = []
for i in shenwan_industry_names:
    a = shenwan_industry(i)
    l = len(a)
    num.append(l)

## 股票总个数
#sum(num)
#3227

shenwan_df['num'] = num
shenwan_df.to_csv(r'./common/shenwan_df.txt', header=True, index=None, sep=' ', mode='a')


shenwan_df




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


