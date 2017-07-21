import xuchuan.ptfopt as pt
import optimizer_tests.stock_test.ptfopt as pt1
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


bounds = {"600150.XSHG": (0, 0.001)}

# to_do_list = list(np.random.choice(equity_funds_list, size=100, replace=False))
to_do_list = ["601099.XSHG", "002594.XSHE", "000423.XSHE", "601390.XSHG", "002195.XSHE", "603000.XSHG", "603885.XSHG",
              "600369.XSHG", "600547.XSHG"]


error_list =['601857.XSHG',
 '601398.XSHG',
 '601939.XSHG',
 '601288.XSHG',
 '601988.XSHG',
 '600028.XSHG',
 '601628.XSHG',
 '601318.XSHG',
 '601088.XSHG',
 '601328.XSHG',
 '600036.XSHG',
 '600016.XSHG',
 '601166.XSHG',
 '601998.XSHG',
 '600000.XSHG',
 '601601.XSHG',
 '600104.XSHG',
 '600030.XSHG',
 '600519.XSHG',
 '601633.XSHG',
 '601818.XSHG',
 '600018.XSHG',
 '601006.XSHG',
 '600837.XSHG',
 '600900.XSHG',
 '000895.XSHE',
 '000001.XSHE',
 '601808.XSHG',
 '000651.XSHE',
 '601668.XSHG',
 '002415.XSHE',
 '600585.XSHG',
 '002594.XSHE',
 '000002.XSHE',
 '601989.XSHG',
 '600887.XSHG',
 '600015.XSHG',
 '000776.XSHE',
 '601336.XSHG',
 '600011.XSHG',
 '000538.XSHE',
 '601766.XSHG',
 '600050.XSHG',
 '600019.XSHG',
 '002024.XSHE',
 '600600.XSHG',
 '601169.XSHG',
 '601800.XSHG',
 '601898.XSHG',
 '000858.XSHE',
 '600999.XSHG',
 '600048.XSHG',
 '601186.XSHG',
 '601390.XSHG',
 '601991.XSHG',
 '600111.XSHG',
 '002241.XSHE',
 '000625.XSHE',
 '600690.XSHG',
 '601238.XSHG',
 '601111.XSHG',
 '600276.XSHG',
 '601899.XSHG',
 '601688.XSHG',
 '600362.XSHG',
 '600031.XSHG',
 '601727.XSHG',
 '002353.XSHE',
 '002236.XSHE',
 '601600.XSHG',
 '600256.XSHG',
 '000063.XSHE',
 '600309.XSHG',
 '600535.XSHG',
 '002304.XSHE',
 '600196.XSHG',
 '600188.XSHG',
 '000157.XSHE',
 '600372.XSHG',
 '600637.XSHG',
 '600795.XSHG',
 '601607.XSHG',
 '600518.XSHG',
 '000039.XSHE',
 '601117.XSHG',
 '000069.XSHE',
 '000338.XSHE',
 '300070.XSHE',
 '601901.XSHG',
 '600703.XSHG',
 '600332.XSHG',
 '600115.XSHG',
 '600010.XSHG',
 '600583.XSHG',
 '601888.XSHG',
 '300027.XSHE',
 '601618.XSHG',
 '600150.XSHG',
 '603993.XSHG',
 '600688.XSHG']


bounds = {"full_list": (0, 0.025)}
order_book_ids = ['000404',
 '550009',
 '550008',
 '400018',
 '519692',
 '420001',
 '166802',
 '000368',
 '160218',
 '000042',
 '519116',
 '160807']
#
# {0: datetime.date(2013, 12, 31),
#  1: datetime.date(2014, 4, 11),
#  2: datetime.date(2014, 7, 17),
#  3: datetime.date(2014, 10, 27),
#  4: datetime.date(2015, 1, 29),
#  5: datetime.date(2015, 5, 12),
#  6: datetime.date(2015, 8, 13),
#  7: datetime.date(2015, 11, 24),
#  8: datetime.date(2016, 3, 3),
#  9: datetime.date(2016, 6, 7),
#  10: datetime.date(2016, 9, 9),
#  11: datetime.date(2016, 12, 21),
#  12: datetime.date(2017, 3, 31),
#  13: datetime.date(2017, 5, 26)}
# #

constraints = {"Hybrid": (0, 0.4), "StockIndex": (0, 0.6)}
optimal_weight = pt1.optimizer(error_list, start_date="2015-8-13", asset_type='stock', method='risk_parity_with_con',
                              iprint=2, disp=True, bnds=bounds)


print(optimal_weight[0])
print(optimal_weight[2])
# print(optimal_weight[3])
# Test shrinkage efficacy
# clean_price = pt.data_process(to_do_list, asset_type="stock", start_date="2016-6-7", windows=132, data_freq="D")
# result_m = pt.cov_shrinkage(clean_price[0])
# print(result_m[0])
# print(result_m[1])

# stock_test_suite1 = {1: error_list}