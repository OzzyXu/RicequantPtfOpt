#import operator
import numpy as np
#import math

###########################################################
##  以下部分是用来取得基金或者股票的分类信息

import rqdatac
rqdatac.init("ricequant", "8ricequant8")


code_lists = ["002861.XSHE", "000670.XSHE", "002579.XSHE"]

length = len(code_lists)
df = pd.DataFrame(index = range(length), columns = ['code', 'type'])
df['code'] =  code_lists

if foo is 'funds':
    for i in range(length):
        df.iloc[i,1] = fund.instruments(code_lists[i]).fund_type

elif foo is 'stocks':
    for i in range(length):
        df.iloc[i,1] = instruments(code_lists[i]).shenwan_industry_name
                #instruments(i).shenwan_industry_code


###########################################################






def get_optimizer_indicators(weight, cov_matrix, type_tag = None):

    weight = np.array(weight)
    # refer to paper formula 2.19 2.20
    production_i = weight * ( cov_matrix.dot(weight) )
    productions = weight.dot(cov_matrix).dot(weight)


## calculate for individual:

    # calculate hightest risk contributions
    HRCs = production_i / productions
    #index, value = max(enumerate(HRCs), key=operator.itemgetter(1))
    HRC, HRC_index = max([(v, i) for i, v in enumerate(HRCs)])

    # calculate Herfindahl
    Herfindahl = np.sum(HRCs**2)

    if type_tag is None:
        return(HRC, HRC_index, Herfindahl)

## calculate for groups:

    # calculate hightest risk contributions
    else:

        df1 = pd.DataFrame(index= range(len(weight)), columns = ['HRCs', 'type'])
        df1['HRCs'] = HRCs
        df1['type'] = type_tag
        productionG_idf1.groupby(['type'])['HRCs'].sum()
        HRCGs = productionG_i / productions
    #index, value = max(enumerate(HRCs), key=operator.itemgetter(1))
        HRCG, HRCG_index = max([(v, i) for i, v in enumerate(HRCGs)])

    # calculate Herfindahl
        Herfindahl_G = np.sum(HRCGs**2)
        return (HRC, HRC_index, Herfindahl, HRCG, HRCG_index, Herfindahl_G)











##  ex

# cov_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]])
# weight = np.array([0.5, 0.35, 0.15])
# weight = [0.5, 0.333, 0.167]

# get_optimizer_indicators(weight , cov_matrix)





