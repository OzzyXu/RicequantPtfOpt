def get_risk_indicators(previous_weight, current_weight, cov_matrix, asset_type):
    """
    Calculate the risk indicators
    :param previous_weight: list or array
    :param current_weight: list or array
    :param cov_matrix: data_frame
    :param asset_type: str
                    'fund' or 'stock'
    :return: list
            HRCs, HRCGs, Herfindahl, turnover_rate
    """

    # change list to array for later process
    # previous_weight_array = np.array(previous_weight)
    current_weight_array = np.array(current_weight)

    # refer to paper formula 2.19 2.20
    production_i = current_weight_array * (cov_matrix.dot(current_weight_array))
    productions = current_weight_array.dot(cov_matrix).dot(current_weight_array)


    # calculate individual's risk contributions
    HRCs = production_i / productions
    # calculate Herfindahl
    Herfindahl = np.sum(HRCs ** 2)



    # calculate group's risk contributions
    df1 = pd.DataFrame(columns=['HRCs', 'type'])
    df1['HRCs'] = HRCs
    if asset_type is 'fund':
        for i in current_weight.index:
            df1.loc[i, 'type'] = fund.instruments(i).fund_type
    elif asset_type is 'stock':
        for i in current_weight.index:
            df1.loc[i, "type"] = rqdatac.instruments(i).shenwan_industry_name

    productionG_i = df1.groupby(['type'])['HRCs'].sum()
    HRCGs = productionG_i

    # calculate group's Herfindahl
    # Herfindahl_G = np.sum(HRCGs ** 2)

    # weight turnover Rate (http://factors.chinascope.com/docs/factors/#turnover)
    # if previous_weight is missing some asset, set the weight to 0

    df2 = pd.DataFrame(columns = ['previous_weight', 'current_weight'])
    df2.current_weight = current_weight
    df2.previous_weight = previous_weight
    df2 = df2.fillna(0)

    turnover_rate = sum(abs(df2.current_weight - df2.previous_weight))/2

    # return_dic = {'individual_RC': HRCs, 'individual_Herfindahl': Herfindahl,
    #               'group_RC': HRCGs, 'group_Herfindahl': Herfindahl_G,
    #               'turnover_rate': turnover_rate}

    return HRCs, HRCGs, Herfindahl, turnover_rate