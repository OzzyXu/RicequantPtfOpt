####################
# fund test
####################


########################################
# !!!! You need to have github structure to run all the code without modificaiton.
########################################
# import all related functions

from optimizer_tests.fund_test.optimizer_fund_test import *



# generate fund_test_suite
fund_test_suite = get_fund_test_suite('2014-01-01')



# run

bigboss = test_fund_opt(fund_test_suite)
bigboss_b_nc = test_fund_opt(fund_test_suite, 1)
bigboss_nb_c = test_fund_opt(fund_test_suite, 2)
bigboss_b_c = test_fund_opt(fund_test_suite, 3)



# to get efficient plots, run get_efficient_plot.py
get_efficient_plots(fund_test_suite, bigboss)
get_efficient_plots(fund_test_suite, bigboss_b_nc)
get_efficient_plots(fund_test_suite, bigboss_nb_c)
get_efficient_plots(fund_test_suite, bigboss_b_c)