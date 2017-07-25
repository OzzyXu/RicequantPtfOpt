
import pandas as pd
import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")



def get_industry_matching(clean_order_book_ids, matching_date, matching_index = '000300.XSHG'):
    """
    if specified industry matching, match unselected industry by matching_index
    :param clean_order_book_ids:
    :param matching_date:
    :param matching_index:
    :return:
    """

    # get clean_order_book_ids industry
    selected_df = pd.DataFrame(index = clean_order_book_ids)
    selected_df['industry'] = [rqdatac.shenwan_instrument_industry(s, matching_date) for s in selected_df.index]
    selected_industry = set(selected_df.industry)

    # get matching_index industry and get unselected industry weight
    index_industry = get_index_component_industry_and_marketcap(matching_index, matching_date)
    match_industry = index_industry.loc[~index_industry.industry.isin(selected_industry)]
    industry_matching_weight =  match_industry.percent

    # get how much weights left for optimizer
    optimizer_total_weight = 1 - sum(industry_matching_weight)

    return optimizer_total_weight, industry_matching_weight




def get_index_component_industry_and_marketcap(matching_index, matching_date):
    """
    get matching_index industry and market_cap
    :param matching_index:
    :param matching_date:
    :return:
    """


    # get index components, industry
    i_c = rqdatac.index_components(matching_index, matching_date)
    matching_index_df = pd.DataFrame(index=i_c)
    matching_index_df['industry'] = [rqdatac.shenwan_instrument_industry(s, matching_date) for s in matching_index_df.index]

    # get index market_cap
    market_cap = rqdatac.get_fundamentals(query(fundamentals.eod_derivative_indicator.market_cap).filter(
        fundamentals.eod_derivative_indicator.stockcode.in_(matching_index_df.index)), entry_date=matching_date)
    market_cap_ = market_cap.loc[market_cap.items[0]].transpose()

    # paste them as one df
    matching_index_cap_df = pd.concat([matching_index_df, market_cap_], axis=1)

    # change the column name
    matching_index_cap_df.columns.values[1] = 'market_cap'

    # calculate each component's percent by its market_cap
    total_market_cap = sum(matching_index_cap_df.market_cap)
    matching_index_cap_df['percent'] = matching_index_cap_df.market_cap/total_market_cap

    # sort them by industry and market_cap
    res = matching_index_cap_df.sort_values(['industry', 'market_cap'])
    return res


