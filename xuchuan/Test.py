import ptfopt as pt
import xuchuan.ptfopt as pt

stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = stock_fund_list
equity_funds_list = ['002832',
 '002901',
 '002341',
 '003176',
 '003634',
 '002621',
 '000916',
 '001416']

optimal_weight = pt.optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')
print(optimal_weight[0])
