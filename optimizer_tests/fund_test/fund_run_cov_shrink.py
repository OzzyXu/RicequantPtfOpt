####################
# fund test
####################


########################################
# !!!! You need to have github structure to run all the code without modificaiton.
########################################
# import all related functions

from optimizer_tests.fund_test.optimizer_fund_test_cov_shrink import *



# generate fund_test_suite
# fund_test_suite = get_fund_test_suite('2014-01-01')
# fund_test_suite_large = get_fund_test_suite('2014-01-01', big = 1)
#len(fund_test_suite_large)


import pickle
# pickle.dump(fund_test_suite, open( "./optimizer_tests/fund_test/fund_test_suite_file/fund_test_suite.p", "wb" ))

## To read it back to use again
fund_test_suite = pickle.load(open( "./optimizer_tests/fund_test/fund_test_suite_file/fund_test_suite.p", "rb" ))





## run

res0_cov_shrink = test_fund_opt(fund_test_suite)
res1_cov_shrink = test_fund_opt(fund_test_suite, 1)
res2_cov_shrink = test_fund_opt(fund_test_suite, 2)
res3_cov_shrink = test_fund_opt(fund_test_suite, 3)



# to get efficient plots, run get_efficient_plot.py
get_efficient_plots(fund_test_suite, res0_cov_shrink, 0)
get_efficient_plots(fund_test_suite, res1_cov_shrink, 1)
get_efficient_plots(fund_test_suite, res2_cov_shrink, 2)
get_efficient_plots(fund_test_suite, res3_cov_shrink, 3)




########################################
# To save all the files for replicating next time

pickle.dump(res0_cov_shrink, open('./optimizer_tests/fund_test/result/normal/save_res/res0_cov_shrink.p', "wb" ))
pickle.dump(res1_cov_shrink, open('./optimizer_tests/fund_test/result/normal/save_res/res1_cov_shrink.p', "wb" ))
pickle.dump(res2_cov_shrink, open('./optimizer_tests/fund_test/result/normal/save_res/res2_cov_shrink.p', "wb" ))
pickle.dump(res3_cov_shrink, open('./optimizer_tests/fund_test/result/normal/save_res/res3_cov_shrink.p', "wb" ))


# To read it back to use again
temp = pickle.load(open('./optimizer_tests/fund_test/result/normal/save_res/res0_cov_shrink.p', "rb" ))

