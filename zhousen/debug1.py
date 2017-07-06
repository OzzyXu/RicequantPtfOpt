bigboss['Hybrid_QDII_StockIndex_3 x 3=9'][0]

len(bigboss['Hybrid_Related_Stock_3 x 4=12'][0])



## xuchuan
fund_test_suite['Hybrid_Related_Stock_3 x 4=12']
##########



start_date = '2014-01-01'

end_date = dt.date.today().strftime("%Y-%m-%d")





order_book_ids = fund_test_suite['Bond_QDII_Related_StockIndex_4 x 3=12']


# missing data
fund.get_nav(order_book_ids, '2014-01-01', '2017-07-03', fields='adjusted_net_value')
get_price('002813.XSHE', '2014-01-01', '2017-07-03', frequency='1d', fields=['close'])



qq1 = period_daily_return_pct_change.loc['2017-06-04':'2017-06-30']
# 20 12
qq1.shape


qq2 = weights[i]['equal_weight']

len(qq2) # 12

qq3 = qq1.multiply(qq2)
qq3
sum(qq3.iloc[0])
sum(qq3.iloc[1])

qq3.sum(axis=1)




fund_test_suite_large['Bond_1 x 100=100']