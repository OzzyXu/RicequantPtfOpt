import numpy as np
import statsmodels as sm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")



def get_industry_matching(clean_order_book_ids, matching_date, matching_index = '000300.XSHG'):




    SHENWAN_INDUSTRY_MAP = {
        "801010.INDX": "农林牧渔",
        "801020.INDX": "采掘",
        "801030.INDX": "化工",
        "801040.INDX": "钢铁",
        "801050.INDX": "有色金属",
        "801080.INDX": "电子",
        "801110.INDX": "家用电器",
        "801120.INDX": "食品饮料",
        "801130.INDX": "纺织服装",
        "801140.INDX": "轻工制造",
        "801150.INDX": "医药生物",
        "801160.INDX": "公用事业",
        "801170.INDX": "交通运输",
        "801180.INDX": "房地产",
        "801200.INDX": "商业贸易",
        "801210.INDX": "休闲服务",
        "801230.INDX": "综合",
        "801710.INDX": "建筑材料",
        "801720.INDX": "建筑装饰",
        "801730.INDX": "电气设备",
        "801740.INDX": "国防军工",
        "801750.INDX": "计算机",
        "801760.INDX": "传媒",
        "801770.INDX": "通信",
        "801780.INDX": "银行",
        "801790.INDX": "非银金融",
        "801880.INDX": "汽车",
        "801890.INDX": "机械设备"
    }

    benchmark_code = "000300.XSHG"  # 沪深300
    date = '20161231'

    i_c = rqdatac.index_components(matching_index, matching_date)
    matchin_index_df = pd.DataFrame(index=i_c)


    benchmark["industry"] = [rqdatac.shenwan_instrument_industry(s, date) for s in benchmark.index]
    market_cap = rqdatac.get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap).filter(
        fundamentals.eod_derivative_indicator.stockcode.in_(benchmark.index)), entry_date=date)

    market_cap_ = market_cap.loc[market_cap.items[0]].transpose()
    result = pd.concat([benchmark, market_cap_], axis=1)
    column_names = result.columns.values
    column_names[1] = "market_cap"
    result.columns = column_names
    result

    result[result.industry == ("801790.INDX", "非银金融")].sort_values("market_cap", ascending=False)




    def get_index_component_and_industry