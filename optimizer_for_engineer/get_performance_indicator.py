def get_performance_indicator(arithmetic_return_of_optimizers):


    arithmetic_return_of_optimizers[0] = 0
    log_return_of_optimizers = np.log(arithmetic_return_of_optimizers + 1)
    annualized_vol = np.sqrt(244) * log_return_of_optimizers.std()
    days_count = len(arithmetic_return_of_optimizers) + 1
    daily_cum_log_return = log_return_of_optimizers.cumsum()
    annualized_cum_return = (daily_cum_log_return[-1] + 1) ** (244 / days_count) - 1

    #mmd = get_maxdrawdown(daily_methods_period_price)



    # get maxdrawdown (refer to https://github.com/ricequant/rqalpha/blob/master/rqalpha/utils/risk.py)
    df_cum = np.exp(np.log1p(arithmetic_return_of_optimizers).cumsum())
    max_return = np.maximum.accumulate(df_cum)
    max_drawdown = ((df_cum - max_return) / max_return).min()
    max_drawdown = abs(max_drawdown)



    return annualized_cum_return, annualized_vol, max_drawdown















def get_maxdrawdown(x):
    # x need to pandas type
    x_cummax = x.cummax()
    cmaxx = x_cummax - x
    cmaxx_p = cmaxx/x_cummax
    mdd = cmaxx_p.max()
    pos1 = [i for i in cmaxx_p.index if cmaxx_p.loc[i]== mdd]

    pos0 =[]
    time0 = [cmaxx_p.index[0]] + pos1
    for i in range(len(pos1)):
        data0 = cmaxx_p.loc[time0[0]:time0[i+1]]
        t = max(data0[(data0 == 0)].index)
        pos0.append(t)

    return mdd, pos0, pos1