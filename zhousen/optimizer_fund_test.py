# created by cc on June 27th



# To run multiple tests, suppress the interactive plots in python first



import matplotlib
matplotlib.use('Agg')


# now import all the necessary modules


import rqdatac
from rqdatac import *
rqdatac.init("ricequant", "8ricequant8")


import datetime as dt
import itertools


import matplotlib.pyplot as plt



# get fund_test_suite ready

fund_test_suite = get_fund_test_suite('2014-01-01')

# now wrap and run

start_date = '2014-01-01'
end_date =  '2017-06-13'
frequency = 132

res_zs = wrap_and_run(fund_test_suite, start_date, end_date, frequency )