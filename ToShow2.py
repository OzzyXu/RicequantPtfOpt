# 05/23/2017 By Chuan Xu @ Ricequant
import rqdatac
import ptfopt_tools as pt
import matplotlib.pyplot as plt


rqdatac.init('ricequant', '8ricequant8')

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']
balanced_funds_list = ['519156', '762001', '160212', '160211', '519983']
fixed_income_funds_list = ['110027', '050011', '519977', '161115', '151002']
index_funds_list = ['100032', '162213', '000311', '090010', '310318']
principal_guaranteed_funds_list = ['000030', '000072', '200016', '000270', '163823']
QDII_list = ['000071', '164705', '110031', '160717', '206006']

first_in_sample_start_time = '2015-01-01'
first_in_sample_end_time = '2015-06-30'
first_out_of_sample_start_time = '2015-07-01'
first_out_of_sample_end_time = '2015-12-31'
x0 = [0.2, 0.2, 0.2, 0.2, 0.2]

[ef_log_barrier_weights, ef_min_variance_weights, ef_min_variance_risk_parity_weights] = \
    pt.portfolio_performance_plot_gen(equity_funds_list,first_in_sample_start_time,first_in_sample_end_time,
                                      ptf_type='equity funds')
plt.show()
[pgf_log_barrier_weights, pgf_min_variance_weights, pgf_min_variance_risk_parity_weights] = \
    pt.portfolio_performance_plot_gen(principal_guaranteed_funds_list,first_in_sample_start_time,
                                      first_in_sample_end_time,ptf_type='principal guaranteed funds')
plt.show()
[QDII_log_barrier_weights, QDII_min_variance_weights, QDII_min_variance_risk_parity_weights] = \
    pt.portfolio_performance_plot_gen(QDII_list,first_in_sample_start_time,first_in_sample_end_time,ptf_type='QDII')
plt.show()
ef_opt_weights = {'log barrier':ef_log_barrier_weights, 'min variance':ef_min_variance_weights,
                  'min variance risk parity':ef_min_variance_risk_parity_weights}
pgf_opt_weights = {'log barrier':pgf_log_barrier_weights, 'min variance':pgf_min_variance_weights,
                  'min variance risk parity':pgf_min_variance_risk_parity_weights}
QDII_opt_weights = {'log barrier':QDII_log_barrier_weights, 'min variance':QDII_min_variance_weights,
                  'min variance risk parity':QDII_min_variance_risk_parity_weights}
pt.portfolio_performance_plot_gen(equity_funds_list, first_out_of_sample_start_time, first_out_of_sample_end_time,
                                  ptf_type='equity funds', weights=ef_opt_weights)
plt.show()
pt.portfolio_performance_plot_gen(principal_guaranteed_funds_list, first_out_of_sample_start_time, first_out_of_sample_end_time,
                                  ptf_type='principal guaranteed funds', weights=pgf_opt_weights)
plt.show()
pt.portfolio_performance_plot_gen(QDII_list, first_out_of_sample_start_time, first_out_of_sample_end_time,
                                  ptf_type='QDII', weights=QDII_opt_weights)
plt.show()
