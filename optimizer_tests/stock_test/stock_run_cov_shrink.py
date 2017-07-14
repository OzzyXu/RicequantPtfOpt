####################
# stock test
####################


########################################
# !!!! You need to have github structure to run all the code without modificaiton.
########################################
# import all related functions


from optimizer_tests.stock_test.optimizer_stock_test_cov_shrink import *




##  generate stock_test_suite
# stock_test_suite = get_stock_test_suite()
# len(stock_test_suite)
# 100

## The upper process takes a while, so we saved the file for reuse

import pickle
# pickle.dump(stock_test_suite, open( "./optimizer_tests/stock_test/stock_test_suite_file/stock_test_suite.p", "wb" ))

## To read it back to use again
stock_test_suite = pickle.load(open( "./optimizer_tests/stock_test/stock_test_suite_file/stock_test_suite.p", "rb" ))




## run

res0_cov_shrink = test_stock_opt(stock_test_suite)
res1_cov_shrink = test_stock_opt(stock_test_suite, 1)
res2_cov_shrink = test_stock_opt(stock_test_suite, 2)
res3_cov_shrink = test_stock_opt(stock_test_suite, 3)




# to get efficient plots, run get_efficient_plot.py
get_efficient_plots(stock_test_suite, res0_cov_shrink, 0)
get_efficient_plots(stock_test_suite, res1_cov_shrink, 1)
get_efficient_plots(stock_test_suite, res2_cov_shrink, 2)
get_efficient_plots(stock_test_suite, res3_cov_shrink, 3)


########################################
# To save all the files for replicating next time

pickle.dump(res0_cov_shrink, open('./optimizer_tests/stock_test/result/cov_shrink/save_res/res0_cov_shrink.p', "wb" ))
pickle.dump(res1_cov_shrink, open('./optimizer_tests/stock_test/result/cov_shrink/save_res/res1_cov_shrink.p', "wb" ))
pickle.dump(res2_cov_shrink, open('./optimizer_tests/stock_test/result/cov_shrink/save_res/res2_cov_shrink.p', "wb" ))
pickle.dump(res3_cov_shrink, open('./optimizer_tests/stock_test/result/cov_shrink/save_res/res3_cov_shrink.p', "wb" ))


# To read it back to use again
temp = pickle.load(open('./optimizer_tests/stock_test/result/cov_shrink/save_res/res0_cov_shrink.p', "rb" ))

