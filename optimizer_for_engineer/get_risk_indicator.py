def get_risk_indicators(previous_weight, current_weight, cov_matrix, asset_type, type_tag = 1):
    """
    To calculate highest rick contributor individually or grouply
    :param weight0: (list). weight at current time t
    :param cov_matrix: (np.matrix). cov_matrix calculated at current time t
    :param asset_type: (str). 'fund' or 'stock'
    :param type_tag: (int). indicator about whether we need group risk contributor or not. 0 means no, 1 means yes
    :return:
    """
    # change list to array for later process
    previous_weight = np.array(previous_weight)
    current_weight = np.array(current_weight)

    # refer to paper formula 2.19 2.20
    production_i = current_weight * (cov_matrix.dot(current_weight))
    productions = current_weight.dot(cov_matrix).dot(current_weight)


    # calculate individual's risk contributions
    HRCs = production_i / productions
    # calculate Herfindahl
    Herfindahl = np.sum(HRCs ** 2)



    # calculate group's risk contributions
    df1 = pd.DataFrame(columns=['HRCs', 'type'])
    df1['HRCs'] = HRCs
    if asset_type is 'fund':
        for i in weight0.index:
            df1.loc[i, 'type'] = fund.instruments(i).fund_type
    elif asset_type is 'stock':
        for i in weight0.index:
            df1.loc[i, "type"] = rqdatac.instruments(i).shenwan_industry_name

    productionG_i = df1.groupby(['type'])['HRCs'].sum()
    HRCGs = productionG_i

    # calculate group's Herfindahl
    Herfindahl_G = np.sum(HRCGs ** 2)

    # weight turnover Rate




    if type_tag == 0:
        return (HRC, HRCs.index(HRC_index), HRC_id, Herfindahl)
    else:
        return (HRC, HRCs.index[HRC_index], HRC_id, Herfindahl, HRCG, HRCGs.index[HRCG_index], Herfindahl_G)
