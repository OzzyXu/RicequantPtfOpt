equity_funds_list1 = ['002832', '002901', '002341', '003176', '001237', '004069', '000307', '003524', '003634', '002621', '000916', '001416']
start_date



for stocks, codes in fund_test_suite.items():
    print(codes)
    if sum([x in equity_funds_list1 for x in codes]) == 12:
        print(stocks)




sum([x in ['202211', '160213', '540007', '240005', '100032', '050002'] for x in ['202211', '160213', '540007', '240005', '100032', '050002'] ])


fund_test_suite.keys()


#fund_test_suite.to_csv('a.csv')

import json
with open('./fund_test_suite1.txt', 'w') as file:
    file.write(json.dumps(fund_test_suite, ensure_ascii=False))



equity_funds_list = ['000387', '000272', '000105', '161211', '160706', '040180', '160127', '320022', '163110']




len(daily_arithmetic_return[0])


len(daily_methods_a_r[j][0].append(daily_methods_a_r[j][1]))


daily_methods_a_r[j]+1

len(daily_methods_a_r[j][1])

len(daily_methods_a_r[j])

len(daily_methods_a_r[j][0])





type(daily_methods_a_r[j][0])

a = np.array(

    [i for x in daily_methods_a_r[j] for i in x]
a
log(a.astype('float64'))

log(a)
np.log(Series(a, dtype='float64'))




q1_key = 'Stock_1*7=7'
fund_test_suite[q1_key]

test_suite1 = ['160127', '320022', '163110', '000082', '540007', '519714', '162107']

equity_funds_list = test_suite1



# this code 8.24  adjusted value 更合理
stock = '162107'

order_book_ids = ['160127', '320022', '163110', '000082', '540007', '519714', '162107']


q2_key = 'Other_1*7=7'

fund_test_suite[q2_key]


equity_funds_list = ['000387', '000272', '000105', '000202', '000138', '000347', '110032']


zs1 = daily_methods_a_r['min_variance']
pp1 = zs1.plot


matplotlib.use()

plt.switch_backend('MacOSX')

plt.switch_backend('Agg')
zs1[-500:].plot()

zs1.plot()


ser1 = daily_cum_log_return[-200:-130]
[ser1[x] for x in ser2.index]

plt.scatter(ser1[0:10].index, ser1[0:10])
ser1[0:10]
-
ser2[0:10]

ser2 = daily_cum_log_return[-200:-170]


plt.scatter(ser1.index, ser1)
plt.scatter(ser2.index, ser2)
plt.show()

range(len(daily_cum_log_return))
plt.plot(range(len(daily_cum_log_return)),daily_cum_log_return)

temp[-150:-140]
# 2016-08-23
# 2016-08-24


daily_cum_log_return[-150:-140]


zs2 = fund.get_nav('162107', '2014-01-01', '2017-06-13',fields='acc_net_value').pct_change()

fund.get_nav('162107', '2016-08-21', '2016-09-03',fields='acc_net_value')

fund.get_nav('162107', '2016-08-21', '2016-09-03')

zs2[abs(zs2) > 0.3]

fund.get_split('162107')

(1-1.8568)/1.8568





# get daily risk free rate
risk_free_rate_tenor = None

if risk_free_rate_tenor is None:
    risk_free_rate = rqdatac.get_yield_curve(start_date, end_date, tenor='0S', country='cn')
elif risk_free_rate_tenor in risk_free_rate_dict:
    risk_free_rate = rqdatac.get_yield_curve(start_date, end_date, tenor=risk_free_rate_tenor, country='cn')

risk_free_rate['Daily'] = pd.Series(np.power(1 + risk_free_rate['0S'], 1 / 365) - 1, index=risk_free_rate.index)


def abc(a, b=1, c=2):

    for i in range(5):
        a = a+1
        if a == 2:
            break
    print(a)
    a = 11
    return  a, -1

c = abc(1)

abc(1,c=3)


def abc1(a, *b):
    return a+b[0] +b[1]



abc1(1,2,3)


time_frame_s

zs1 = daily_methods_a_r['risk_parity']
zs1[0:70]
zs1[0:70].index


len(zs1)
849
len(np.unique(zs1.index))

849

time_frame = {}
time_frame_e = {}
for i in range(0, count):
    time_frame_s[i] = trading_dates[i * tr_frequency]
    time_frame_e[i] = trading_dates[(i + 1) * tr_frequency - 1]
time_frame_s[count] = trading_dates[count * tr_frequency]
time_frame_e[count] = end_date



zs2 = weights[0]
type(zs2)
zs2['risk_parity'].index


for i in zs2['risk_parity'].index:
    print(i)


t1=np.array([1,2])

t2 = np.array([[1,2],[3,4]])

t1*(t2.dot(t1))

t1.dot(t2).dot(t1)



###################


kk = 'Bond_Hybrid_Related_Stock_4 x 1=4'

order_book_ids = fund_test_suite[kk]



dict1 = {}
dict1.append({1:2})

df0 = pd.DataFrame(columns = 'type')