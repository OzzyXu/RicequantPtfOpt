
start_date = '2014-01-01'
end_date= '2017-05-31'
method = 'risk_parity_with_con'
asset_type = 'stock'

tr_frequency = 66
order_book_ids =stock_test_suite[0]

trading_date_s = get_previous_trading_date(start_date)
trading_date_e = get_previous_trading_date(end_date)
trading_dates = get_trading_dates(trading_date_s, trading_date_e)
time_len = len(trading_dates)
count = floor(time_len / tr_frequency)

opt_res = {}

time_frame = {}
for i in range(0, count + 1):
    time_frame[i] = trading_dates[i * tr_frequency]
time_frame[count + 1] = trading_dates[-1]




i = 6

opt_res[i] = optimizer(order_book_ids, start_date=time_frame[i], asset_type=asset_type, method=method,
                                   bnds={'full_list': (0, 0.025)}, cons = 1, fun_tol=10**-8, iprint = 2, disp = True)









#
#
# time_frame
# {0: datetime.date(2013, 12, 31),
#  1: datetime.date(2014, 4, 11),
#  2: datetime.date(2014, 7, 17),
#  3: datetime.date(2014, 10, 27),
#  4: datetime.date(2015, 1, 29),
#  5: datetime.date(2015, 5, 12),
#  6: datetime.date(2015, 8, 13),
#  7: datetime.date(2015, 11, 24),
#  8: datetime.date(2016, 3, 3),
#  9: datetime.date(2016, 6, 7),
#  10: datetime.date(2016, 9, 9),
#  11: datetime.date(2016, 12, 21),
#  12: datetime.date(2017, 3, 31),
#  13: datetime.date(2017, 5, 26)}