def get_optimizer_indicators(weight0, cov_matrix, asset_type, type_tag=1):
    """
    To calculate highest rick contributor individually or grouply
    :param weight0: (list). weight at current time t
    :param cov_matrix: (np.matrix). cov_matrix calculated at current time t
    :param asset_type: (str). 'fund' or 'stock'
    :param type_tag: (int). indicator about whether we need group risk contributor or not. 0 means no, 1 means yes
    :return:
    """
    weight = np.array(weight0)
    # refer to paper formula 2.19 2.20
    production_i = weight * (cov_matrix.dot(weight))
    productions = weight.dot(cov_matrix).dot(weight)

    ## calculate for individual:
    # calculate hightest risk contributions
    HRCs = production_i / productions
    # index, value = max(enumerate(HRCs), key=operator.itemgetter(1))
    HRC, HRC_index = max([(v, i) for i, v in enumerate(HRCs)])

    # calculate Herfindahl
    Herfindahl = np.sum(HRCs ** 2)


    ## calculate for groups:
    # calculate hightest risk contributions

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
    HRCG, HRCG_index = max([(v, i) for i, v in enumerate(HRCGs)])

    # calculate Herfindahl
    Herfindahl_G = np.sum(HRCGs ** 2)

    ## get asset code type
    HRC_id = HRCs.index[HRC_index]
    df1.loc[HRC_id]

    if type_tag == 0:
        return (HRC, HRCs.index(HRC_index), HRC_id, Herfindahl)
    else:
        return (HRC, HRCs.index[HRC_index], HRC_id, Herfindahl, HRCG, HRCGs.index[HRCG_index], Herfindahl_G)
