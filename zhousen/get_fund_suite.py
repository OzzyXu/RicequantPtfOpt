def get_fund_test_suite(before_date = dt.date.today().strftime("%Y-%m-%d")):
    fund_pool = {}
    combo = []
    fund_test_suite = {}

    all_fund = fund.all_instruments(date=before_date)[['order_book_id', 'listed_date', 'symbol', 'fund_type']]
    #fund_types = np.unique(all_fund.fund_type)

    # currently we ruled out Other and Money type
    fund_types = ['Bond', 'BondIndex', 'Hybrid', 'QDII', 'Related', 'Stock', 'StockIndex']



    len_fund_types = len(fund_types)


    # get a dictionary of all_fund according to fund_types
    for i in fund_types:
        fund_pool[i] = (all_fund[all_fund['fund_type'] == i]['order_book_id'].values[:])


    # get all possible combinations of fund_types
    for j in range(1, len_fund_types + 1):
        for subset in itertools.combinations(fund_types, j):
            combo.append(subset)
            #print(subset)

    # each_fund_num = {1: [5, 6, 7, 8], 2: [3,4,5,6], 3: [2,3,4], 4:[1,2,3], 5:[1,2], 6:[1,2], 7:[1]}

    each_fund_num = {1: [ 6, 7, 8], 2: [3,4,5,6], 3: [2,3,4], 4:[2,3], 5:[2], 6:[1,2], 7:[1]}
        #, 8:[1], 9:[1]}

    for k in combo:
        len_combo = len(k)
        for l in each_fund_num[len_combo]:
            temp = [a for x in k for a in fund_pool[x][0:l]]
            fund_test_suite['_'.join(k)+'_'+str(len_combo)+' x '+str(l)+'='+str(len(temp))] = temp



    return fund_test_suite




######### example:
# fund_test_suite = get_fund_test_suite('2014-01-01')
# len(fund_test_suite)
# 379



# 316




