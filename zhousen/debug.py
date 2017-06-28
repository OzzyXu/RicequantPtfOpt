equity_funds_list1 = ['002832', '002901', '002341', '003176', '001237', '004069', '000307', '003524', '003634', '002621', '000916', '001416']
start_date



for stocks, codes in fund_test_suite.items():
    print(codes)
    if sum([x in equity_funds_list1 for x in codes]) == 12:
        print(stocks)




sum([x in ['202211', '160213', '540007', '240005', '100032', '050002'] for x in ['202211', '160213', '540007', '240005', '100032', '050002'] ])


fund_test_suite.keys()


#fund_test_suite.to_csv('a.csv')

import json
with open('./fund_test_suite.txt', 'w') as file:
    file.write(json.dumps(fund_test_suite, ensure_ascii=False))



equity_funds_list = ['000387', '000272', '000105', '161211', '160706', '040180', '160127', '320022', '163110']
