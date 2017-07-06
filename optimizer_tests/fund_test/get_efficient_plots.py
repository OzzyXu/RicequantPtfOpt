def get_efficient_plots(fund_test_suite, bigboss):
    """
    To generate the plot of all test results, x-axis is annualized_vol, y-axis is annualized return
    :param fund_test_suite:
    :param bigboss (dic): a dictionary stored all test results
    :return:
    """
    annualized_vol = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "risk_parity_with_con": []}
    annualized_return = {'equal_weight': [], 'min_variance': [], 'risk_parity': [], "risk_parity_with_con": []}


    for j in ['equal_weight', 'min_variance', 'risk_parity', "risk_parity_with_con"]:
        for i in fund_test_suite.keys():
            annualized_vol[j].append(bigboss[i][2][j])
            annualized_return[j].append(bigboss[i][1][j])

    #str1 = 'a'

        p1 = plt.plot(annualized_vol[j], annualized_return[j], 'o', label = j)
        plt.legend()
        plt.xlabel('annualized_vol')
        plt.ylabel('annualized_return')
        plt.show()

    return p1

