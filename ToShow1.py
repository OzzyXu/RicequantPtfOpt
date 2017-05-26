# 05/23/2017 By Chuan Xu @ Ricequant
import ptfopt_tools as pt
import rqdatac
import matplotlib.pyplot as plt
import numpy as np

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']
balanced_funds_list = ['519156', '762001', '160212', '160211', '519983']
fixed_income_funds_list = ['110027', '050011', '519977', '161115', '151002']
index_funds_list = ['100032', '162213', '000311', '090010', '310318']
principal_guaranteed_funds_list = ['000030', '000072', '200016', '000270', '163823']
QDII_list = ['000071', '164705', '110031', '160717', '206006']
# dict = {'equity_fund': equity_funds_list, 'balanced_fund': balanced_funds_list,
#         'fixed_income_fund': fixed_income_funds_list, 'index_fund': index_funds_list,
#         'principal_guaranteed_fund': principal_guaranteed_funds_list, 'QDII': QDII_list}

rqdatac.init('ricequant', '8ricequant8')

first_sample_start_time = '2014-01-01'
first_sample_end_time = '2014-06-30'
first_in_sample_equity_fund_portfolio = pt.funds_data_input(equity_list=equity_funds_list,start_date=first_sample_start_time,
                                         end_date=first_sample_end_time)
first_in_sample_principal_guaranteed_fund_portfolio = pt.funds_data_input(equity_list=principal_guaranteed_funds_list,
                                                                     start_date=first_sample_start_time,
                                                                     end_date=first_sample_end_time)
first_in_sample_QDII_portfolio = pt.funds_data_input(equity_list=QDII_list,start_date=first_sample_start_time,
                                               end_date=first_sample_end_time)

first_in_sample_equity_fund_raw_cov = pt.raw_covariance_cal(first_in_sample_equity_fund_portfolio)
first_in_sample_principal_guaranteed_fund_raw_cov = pt.raw_covariance_cal(first_in_sample_principal_guaranteed_fund_portfolio)
first_in_sample_QDII_raw_cov = pt.raw_covariance_cal(first_in_sample_QDII_portfolio)

# First in sample equity funds portfolio performance figure
first_in_sample_log_barrier_risk_parity_equity_fund_opt_ptf = \
        pt.log_barrier_risk_parity_optimizer(first_in_sample_equity_fund_raw_cov)
first_in_sample_min_variance_equity_fund_opt_ptf = \
        pt.min_variance_optimizer(first_in_sample_equity_fund_raw_cov)
first_in_sample_min_variance_risk_parity_equity_fund_opt_ptf  = \
        pt.min_variance_risk_parity_optimizer(first_in_sample_equity_fund_raw_cov)

first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_equity_fund_portfolio,
                                     first_in_sample_log_barrier_risk_parity_equity_fund_opt_ptf[0],
                                     first_in_sample_equity_fund_raw_cov)
first_in_sample_min_variance_equity_fund_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_equity_fund_portfolio,
                                     first_in_sample_min_variance_equity_fund_opt_ptf[0],
                                     first_in_sample_equity_fund_raw_cov)
first_in_sample_min_variance_risk_parity_equity_fund_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_equity_fund_portfolio,
                                     first_in_sample_min_variance_risk_parity_equity_fund_opt_ptf[0],
                                     first_in_sample_equity_fund_raw_cov)
first_in_sample_equal_weights_equity_fund_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_equity_fund_portfolio, x0, first_in_sample_equity_fund_raw_cov)

fig1 = plt.figure(1)
str1 = """Equaity weights portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_equal_weights_equity_fund_ptf_perf[1],first_in_sample_equal_weights_equity_fund_ptf_perf[2],
       ','.join(map(str, x0)), ','.join(map(str,np.round(first_in_sample_equal_weights_equity_fund_ptf_perf[3], decimals=4))))
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_min_variance_equity_fund_ptf_perf[1],first_in_sample_min_variance_equity_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_equity_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_equity_fund_ptf_perf[3],decimals=4))))
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf[1],first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_log_barrier_risk_parity_equity_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf[3],decimals=4))))
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s].""" % \
      (first_in_sample_min_variance_risk_parity_equity_fund_ptf_perf[1],first_in_sample_min_variance_risk_parity_equity_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_risk_parity_equity_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_risk_parity_equity_fund_ptf_perf[3],decimals=4))))
plt.figtext(0.12,0.03,str1+str2+str3+str4)
plt.title('%s to %s In sample equity funds portfolio performance' % (first_in_sample_equity_fund_portfolio.index[0].date(), first_in_sample_equity_fund_portfolio.index[-1].date()))
ax = fig1.add_subplot(111)
plt.subplots_adjust(bottom=0.18)
ax.plot(first_in_sample_equity_fund_portfolio.index, first_in_sample_equal_weights_equity_fund_ptf_perf[0],
         label='Equal weights')
ax.plot(first_in_sample_equity_fund_portfolio.index, first_in_sample_min_variance_equity_fund_ptf_perf[0],
         label='Min vairance')
ax.plot(first_in_sample_equity_fund_portfolio.index, first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf[0],
         label='Log barrier risk parity')
ax.plot(first_in_sample_equity_fund_portfolio.index, first_in_sample_min_variance_risk_parity_equity_fund_ptf_perf[0],
         label='Min variance risk parity')
ax.legend()

# First in sample principal guaranteed funds portfolio performance figure
first_in_sample_log_barrier_risk_parity_PG_fund_opt_ptf = \
        pt.log_barrier_risk_parity_optimizer(first_in_sample_principal_guaranteed_fund_raw_cov)
first_in_sample_min_variance_PG_fund_opt_ptf = \
        pt.min_variance_optimizer(first_in_sample_principal_guaranteed_fund_raw_cov)
first_in_sample_min_variance_risk_parity_PG_fund_opt_ptf  = \
        pt.min_variance_risk_parity_optimizer(first_in_sample_principal_guaranteed_fund_raw_cov)

first_in_sample_log_barrier_risk_parity_PG_fund_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_principal_guaranteed_fund_portfolio,
                                     first_in_sample_log_barrier_risk_parity_PG_fund_opt_ptf[0],
                                     first_in_sample_principal_guaranteed_fund_raw_cov)
first_in_sample_min_variance_PG_fund_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_principal_guaranteed_fund_portfolio,
                                     first_in_sample_min_variance_PG_fund_opt_ptf[0],
                                     first_in_sample_principal_guaranteed_fund_raw_cov)
first_in_sample_min_variance_risk_parity_PG_fund_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_principal_guaranteed_fund_portfolio,
                                     first_in_sample_min_variance_risk_parity_PG_fund_opt_ptf[0],
                                     first_in_sample_principal_guaranteed_fund_raw_cov)
first_in_sample_equal_weights_PG_fund_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_principal_guaranteed_fund_portfolio, x0,
                                     first_in_sample_principal_guaranteed_fund_raw_cov)

fig2 = plt.figure(2)
str1 = """Equaity weights portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_equal_weights_PG_fund_ptf_perf[1],first_in_sample_equal_weights_PG_fund_ptf_perf[2],
       ','.join(map(str, x0)), ','.join(map(str,np.round(first_in_sample_equal_weights_PG_fund_ptf_perf[3], decimals=4))))
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_min_variance_PG_fund_ptf_perf[1],first_in_sample_min_variance_PG_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_PG_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_PG_fund_ptf_perf[3],decimals=4))))
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_log_barrier_risk_parity_PG_fund_ptf_perf[1],first_in_sample_log_barrier_risk_parity_PG_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_log_barrier_risk_parity_PG_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_log_barrier_risk_parity_PG_fund_ptf_perf[3],decimals=4))))
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s].""" % \
      (first_in_sample_min_variance_risk_parity_PG_fund_ptf_perf[1],first_in_sample_min_variance_risk_parity_PG_fund_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_risk_parity_PG_fund_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_risk_parity_PG_fund_ptf_perf[3],decimals=4))))
plt.figtext(0.12,0.03,str1+str2+str3+str4)
plt.title('01-01-2014 to 06-30-2014 In sample principal guaranteed funds portfolio performance')
ax = fig2.add_subplot(111)
plt.subplots_adjust(bottom=0.18)
ax.plot(first_in_sample_principal_guaranteed_fund_portfolio.index, first_in_sample_equal_weights_PG_fund_ptf_perf[0],
         label='Equal weights')
ax.plot(first_in_sample_principal_guaranteed_fund_portfolio.index, first_in_sample_min_variance_PG_fund_ptf_perf[0],
         label='Min vairance')
ax.plot(first_in_sample_principal_guaranteed_fund_portfolio.index, first_in_sample_log_barrier_risk_parity_equity_fund_ptf_perf[0],
         label='Log barrier risk parity')
ax.plot(first_in_sample_principal_guaranteed_fund_portfolio.index, first_in_sample_log_barrier_risk_parity_PG_fund_ptf_perf[0],
         label='Min variance risk parity')
ax.legend()

# First in sample QDII portfolio performance figure
first_in_sample_log_barrier_risk_parity_QDII_opt_ptf = \
        pt.log_barrier_risk_parity_optimizer(first_in_sample_QDII_raw_cov)
first_in_sample_min_variance_QDII_opt_ptf = \
        pt.min_variance_optimizer(first_in_sample_QDII_raw_cov)
first_in_sample_min_variance_risk_parity_QDII_opt_ptf  = \
        pt.min_variance_risk_parity_optimizer(first_in_sample_QDII_raw_cov)

first_in_sample_log_barrier_risk_parity_QDII_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_QDII_portfolio,
                                     first_in_sample_log_barrier_risk_parity_QDII_opt_ptf[0], first_in_sample_QDII_raw_cov)
first_in_sample_min_variance_QDII_ptf_perf = \
        pt.portfolio_performance_gen(first_in_sample_QDII_portfolio,
                                     first_in_sample_min_variance_QDII_opt_ptf[0], first_in_sample_QDII_raw_cov)
first_in_sample_min_variance_risk_parity_QDII_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_QDII_portfolio,
                                     first_in_sample_min_variance_risk_parity_QDII_opt_ptf[0], first_in_sample_QDII_raw_cov)
first_in_sample_equal_weights_QDII_ptf_perf  = \
        pt.portfolio_performance_gen(first_in_sample_QDII_portfolio, x0, first_in_sample_QDII_raw_cov)

fig3 = plt.figure(3)
str1 = """Equaity weights portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_equal_weights_QDII_ptf_perf[1],first_in_sample_equal_weights_QDII_ptf_perf[2],
       ','.join(map(str, x0)), ','.join(map(str,np.round(first_in_sample_equal_weights_QDII_ptf_perf[3], decimals=4))))
str2 = """Min variance portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_min_variance_QDII_ptf_perf[1],first_in_sample_min_variance_QDII_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_QDII_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_QDII_ptf_perf[3],decimals=4))))
str3 = """Log barrier risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s]. \n""" % \
      (first_in_sample_log_barrier_risk_parity_QDII_ptf_perf[1],first_in_sample_log_barrier_risk_parity_QDII_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_log_barrier_risk_parity_QDII_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_log_barrier_risk_parity_QDII_ptf_perf[3],decimals=4))))
str4 = """Min variance risk parity portfolio: r = %f, $\sigma$ = %f, weights = [%s], risk contribution = [%s].""" % \
      (first_in_sample_min_variance_risk_parity_QDII_ptf_perf[1],first_in_sample_min_variance_risk_parity_QDII_ptf_perf[2],
       ','.join(map(str, np.round(first_in_sample_min_variance_risk_parity_QDII_opt_ptf[0],decimals=4))),
       ','.join(map(str,np.round(first_in_sample_min_variance_risk_parity_QDII_ptf_perf[3],decimals=4))))
plt.figtext(0.12,0.03,str1+str2+str3+str4)
plt.title('01-01-2014 to 06-30-2014 In sample QDII portfolio performance')
ax = fig3.add_subplot(111)
plt.subplots_adjust(bottom=0.18)
ax.plot(first_in_sample_QDII_portfolio.index, first_in_sample_equal_weights_QDII_ptf_perf[0],
         label='Equal weights')
ax.plot(first_in_sample_QDII_portfolio.index, first_in_sample_min_variance_QDII_ptf_perf[0],
         label='Min vairance')
ax.plot(first_in_sample_QDII_portfolio.index, first_in_sample_log_barrier_risk_parity_QDII_ptf_perf[0],
         label='Log barrier risk parity')
ax.plot(first_in_sample_QDII_portfolio.index, first_in_sample_min_variance_risk_parity_QDII_ptf_perf[0],
         label='Min variance risk parity')
ax.legend()

plt.show()