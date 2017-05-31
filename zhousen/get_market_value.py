# 以下函数用来求股票的市值  （市值 = 总股本 x 选定日期收盘价）
# input:  stock_list 可以是str or str_list,  start_date, end_date
# output: str_list 会返回 pandas data frame
#         str 返回 series

def get_market_value(stock_list, start_date = '2017-05-25', end_date = '2017-05-26'):
    p1 = get_price(stock_list, start_date = start_date, end_date = end_date, fields = 'close')
    s1 = get_shares(stock_list, start_date, end_date, fields = 'total')
    if isinstance(stock_list, str):
        return(p1*s1)
    else:
        market_value = []
        for i in range(len(stock_list)):
            market_value.append(p1.iloc[:,i] * s1.iloc[:,i])

        res = np.transpose(pd.DataFrame(market_value))
        return(res)



# ex:
# stock_list =[ '601998.XSHG','603323.XSHG']
# get_market_value(stock_list)

# stock_list = '601998.XSHG'
# get_market_value(stock_list)