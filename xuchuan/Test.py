import xuchuan.ptfopt as pt
import numpy as np

stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = stock_fund_list
equity_funds_list = ['002832',
 '002901',
 '002341',
 '003176',
 '002621',
 '000916',
 '001416']

# equity_funds_list = ['540006', '163110', '450009', '160716', '162213']

# N*1 vector
investors_views = np.matrix([[0.0001], [0.0002]])
# # K*N matrix
# investors_views_indicate_M = np.matrix([[0, -1, 0, 1, 0],
#                                        [-1, 0, 0, 0, 1]])
# K*N matrix
investors_views_indicate_M = np.matrix([[0, -1, 0, 1, 0, 0, 0],
                                       [-1, 0, 0, 0, 0, 0, 1]])
# A list with K elements
confidence_of_views_list = [0.5, 0.5]

res = pt.black_litterman_prep(equity_funds_list, "2017-06-01", investors_views, investors_views_indicate_M,
                           asset_type='fund')
expected_return_list = res[0]
expected_return_covar_M = res[1]
risk_aversion_c = res[2]
expected_return_list = np.matrix([0.01]*7).transpose()

optimal_weight = pt.optimizer(equity_funds_list, start_date="2017-06-01", asset_type='fund', method='all',
                              expected_return=expected_return_list, expected_return_covar=expected_return_covar_M,
                              risk_aversion_coefficient=risk_aversion_c)
print(optimal_weight[0])
print(optimal_weight[1])
