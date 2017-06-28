## get stock arithmetic returns, log returns





start_date = '2016-05-23'

end_date = '2017-05-26'

sample_list = ['002214.XSHE',"002388.XSHE"]

sample_list = ['002214.XSHE']


a = pt.TestPortfolio(['002214.XSHE',"002388.XSHE"], 'stocks')


a = pt.TestPortfolio('002214.XSHE', 'stocks')

a.perf_update(1, '2017-05-23', '2017-05-26')

a.annualized_return



equity_list = ['002813.XSHE','002214.XSHE']


equity_list = ['002214.XSHE']

equity_list = ['002214.XSHE',"002388.XSHE"]


weights = [0.5, 0.5]

weights = 1


a = pt.TestPortfolio(['002214.XSHE'], 'stocks')
a.perf_update(1, '2017-05-23', '2017-05-26')




daily_arithmetic_return = None
daily_cum_log_return = None


period_prices = rqdatac.get_price(sample_list, start_date, end_date, frequency='1d', fields='close')

#period_daily_return_pct_change = period_prices.pct_change()[1:]
period_daily_return_pct_change = pd.DataFrame(period_prices.pct_change())[1:]
new_daily_arithmetic_return = period_daily_return_pct_change.multiply(weights).sum(axis=1)
if daily_arithmetic_return is None:
    daily_arithmetic_return = new_daily_arithmetic_return
else:
    daily_arithmetic_return = daily_arithmetic_return.append(new_daily_arithmetic_return)

new_daily_cum_log_return = np.log(new_daily_arithmetic_return + 1).cumsum()

if daily_cum_log_return is None:
    daily_cum_log_return = new_daily_cum_log_return
else:
    daily_cum_log_return = daily_cum_log_return.append(
        new_daily_cum_log_return + daily_cum_log_return[-1])

temp = np.log(daily_arithmetic_return + 1)
annualized_vol = sqrt(244) * temp.std()
days_count = (daily_arithmetic_return.index[-1] - daily_arithmetic_return.index[0]).days
annualized_return = (daily_cum_log_return[-1] + 1) ** (365 / days_count) - 1


