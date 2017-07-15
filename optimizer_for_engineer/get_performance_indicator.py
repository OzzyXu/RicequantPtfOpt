def get_performance_indicator(asset_daily_return):
    










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