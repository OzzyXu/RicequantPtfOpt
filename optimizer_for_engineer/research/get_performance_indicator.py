def get_performance_indicator(arithmetic_return_of_optimizers):


    arithmetic_return_of_optimizers[0] = 0
    log_return_of_optimizers = np.log(arithmetic_return_of_optimizers + 1)
    annualized_vol = np.sqrt(244) * log_return_of_optimizers.std()
    days_count = len(arithmetic_return_of_optimizers) + 1
    daily_cum_log_return = log_return_of_optimizers.cumsum()
    annualized_cum_return = (daily_cum_log_return[-1] + 1) ** (244 / days_count) - 1


    # get maxdrawdown (refer to https://github.com/ricequant/rqalpha/blob/master/rqalpha/utils/risk.py)
    df_cum = np.exp(np.log1p(arithmetic_return_of_optimizers).cumsum())
    max_return = np.maximum.accumulate(df_cum)
    max_drawdown = ((df_cum - max_return) / max_return).min()
    max_drawdown = abs(max_drawdown)



    return annualized_cum_return, annualized_vol, max_drawdown

