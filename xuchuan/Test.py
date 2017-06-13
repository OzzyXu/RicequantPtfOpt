import ptfopt as pt

stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = stock_fund_list
equity_funds_list = ['540006', '163110', '450009', '160716', '162213']

optimal_weight = pt.optimizer(equity_funds_list, start_date='2015-01-01', equity_type='fund', method='all')
print(optimal_weight[0])