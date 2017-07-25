
import xuchuan.ptfopt as pt


order_book_ids = ['161826',
                  '150134',
                  '000404',
                  '550009',
                  '161116',
                  '118001',
                  '540006',
                  '000309']


order_book_ids = stock_test_suite[0]

constraints = {"Hybrid": (0, 0.4), "StockIndex": (0, 0.6)}
bounds = {"full_list": (0, 0.3)}

method ='risk_parity_with_con'
method = 'mean_variance'
optimal_weight = pt.optimizer(order_book_ids, start_date="2014-1-1", asset_type='fund', method=method,
                              iprint=1, disp=False, bnds=bounds)


print(optimal_weight[0])
print(optimal_weight[2])
print(optimal_weight[3])