import xuchuan.ptfopt as pt
import numpy as np
import rqdatac

rqdatac.init("ricequant", "8ricequant8")


stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = stock_fund_list
equity_funds_list = ['000118',
 '160810',
 '000052',
 '020029',
 '180030',
 '270045',
 '160622',
 '519683',
 '590009',
 '000355',
 '000316',
 '000207',
 '686868',
 '630109',
 '000186',
 '288102',
 '162299',
 '261102',
 '020020',
 '519666',
 '040022',
 '070020',
 '240003',
 '519976',
 '519660',
 '040010',
 '161693',
 '020033',
 '000281',
 '519162',
 '110008',
 '163806',
 '000081',
 '470059',
 '253020',
 '160915',
 '166010',
 '161902',
 '519111',
 '110027',
 '000015',
 '161716',
 '470058',
 '000148',
 '050023',
 '340009',
 '630107',
 '000084',
 '206003',
 '450018',
 '000014',
 '000252',
 '690206',
 '000205',
 '217203',
 '000170',
 '000054',
 '160618',
 '000236',
 '519723',
 '000091',
 '519519',
 '000356',
 '000007',
 '000406',
 '371120',
 '020012',
 '371020',
 '519731',
 '217011',
 '485014',
 '000386',
 '000152',
 '100066',
 '519676',
 '310378',
 '531021',
 '040023',
 '161216',
 '166003',
 '582001',
 '000396',
 '000016',
 '519078',
 '161624',
 '092002',
 '163003',
 '630003',
 '000128',
 '164808',
 '700005',
 '550019',
 '573003',
 '530020',
 '233005',
 '380005',
 '202102',
 '750003',
 '000139',
 '206004']

equity_funds_list = ['000181', '000386', '000069', '160123', '510080', '160124', '460010']

# equity_funds_list = ['540006', '163110', '450009', '160716', '162213']

# equity_funds_list = rqdatac.index_components("000300.XSHG", "2016-12-31")

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

# res = pt.black_litterman_prep(equity_funds_list, "2017-06-01", investors_views, investors_views_indicate_M,
#                            asset_type='fund', data_freq="M", windows=30)
# print(type(res[0]), type(res[1]), type(res[2]), type(res[3]))
# print(res[0], res[1], res[2], res[3])
# expected_return_list = res[0]
# expected_return_covar_M = res[1]
# risk_aversion_c = res[2]
# expected_return_list = np.matrix([0.01]*7).transpose()

constraints = {"国防军工": (0, 0.005)}
bounds = {"600150.XSHG": (0, 0.001)}

# to_do_list = list(np.random.choice(equity_funds_list, size=100, replace=False))
to_do_list = ["601099.XSHG", "002594.XSHE", "000423.XSHE", "601390.XSHG", "002195.XSHE", "603000.XSHG", "603885.XSHG",
              "600369.XSHG", "600547.XSHG"]

# optimal_weight = pt.optimizer(to_do_list, start_date="2017-6-1", asset_type='stock', method='all')
# print(optimal_weight[0])
# print(optimal_weight[2])

# Test shrinkage efficacy
clean_price = pt.data_process(to_do_list, asset_type="stock", start_date="2016-6-7", windows=132, data_freq="D")
result_m = pt.cov_shrinkage(clean_price[0])
print(result_m[0])
print(result_m[1])