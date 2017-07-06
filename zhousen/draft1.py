
import xuchuan.ptfopt as pt










fund_types
try_suite = list(fund_pool['Bond'])[0:4] + list(fund_pool['Stock'])[0:4]
try_suite


order_book_ids = try_suite
start_date = '2017-06-19'
asset_type = 'fund'
method = 'risk_parity'

method = 'all'
equity_funds_list = ['002832', '002901', ]



optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')





'plead work1'

'dsfa'


import ptfopt_zs as pt

stock_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Stock.txt").read()
stock_fund_list = stock_fund_list.splitlines()
# hybrid_fund_list = open("C:\\Users\\xuchu\\Dropbox\\RQ\\Test_Result\\Fund_test_suite\\Hybrid.txt").read()
# hybrid_fund_list = hybrid_fund_list.splitlines()

equity_funds_list = ['002832',
['002901',
 '002341',
 '003176',
 '003634',
 '002621',
 '000916',
 '001416']

equity_funds_list = ['540006', '163110', '450009', '160716', '162213']





equity_funds_list = ['002832',
 '002901',
 '002341',
 '003176',
 '003634',
 '002621',
 '000916',
 '001416']

a1=optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')

a1[0]
a2 = a1[1].head()
a2.pct_change()

import ptfopt_zs as pt
pt.optimizer(equity_funds_list, start_date='2017-06-19', asset_type='fund', method='all')



optimal_weight[0]


print(optimal_weight[0])






index_name = "000300.XSHG"
first_period_s = '2014-01-01'
first_period_e = '2014-06-30'
second_period_s = '2014-07-01'
second_period_e = '2014-12-30'
third_period_s = '2015-01-01'
third_period_e = '2015-06-30'
fourth_period_s = '2015-07-01'
fourth_period_e = '2015-12-30'
fifth_period_s = '2016-01-01'
fifth_period_e = '2016-06-30'
sixth_period_s = '2016-07-01'
sixth_period_e = '2016-12-30'
seventh_period_s = '2017-01-01'
seventh_period_e = '2017-05-20'

portfolio1 = rqdatac.index_weights(index_name, second_period_s)
equity_list1 = list(portfolio1.index)
portfolio2 = rqdatac.index_weights(index_name, third_period_s)
equity_list2 = list(portfolio2.index)
portfolio3 = rqdatac.index_weights(index_name, fourth_period_s)
ty_list+elimination_list))
output_res1.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20140701.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list2, second_period_s, second_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio2.loc[elimination_list])
start_time2 = time.time()
optimal_weights = list(equity_fund_portfolio_min_variance.min_variance_optimizer())
print('MV optimizer running time 2: %s' % (time.time()-start_time2))
optimal_weights = [x*(1-sum(inherited_holdings_weights)) for x in optimal_weights]
weights = optimal_weights+inherited_holdings_weights
output_res2 = pd.Series(weights, index=(equity_fund_portfolio_min_variance.clean_equity_list+elimination_list))
output_res2.to_csv(path='C:\\Users\\xuchu\\Dropbox\\RQ\\Result\\MV20150101.csv')

equity_fund_portfolio_min_variance.data_preprocessing(equity_list3, third_period_s, third_period_e)
elimination_list = equity_fund_portfolio_min_variance.kickout_list+equity_fund_portfolio_min_variance.st_list + \
                   equity_fund_portfolio_min_variance.suspended_list
inherited_holdings_weights = list(portfolio3.loc[elimination_list])


###########################################



function[c, p] = real_option_price(z)

% z is strike price
% This function is used to generate call and put price
% Assumed three log normal adding each other
% refer to Table 1 Case 1 in page 5 in Feng

p1 = 0.1076;
v1 = 7.3580;
sigma1 = 0.1544;

p2 = 0.3350;
v2 = 7.5154;
sigma2 = 0.0769;

p3 = 0.5574;
v3 = 7.6045;
sigma3 = 0.0410;

St = 1920.24;
r = 0.003;
delta = 0.021;
dT = 136 / 365;



% a1 = p1*exp(v1+(sigma1^2)/2)+p2*exp(v2+sigma2^2/2)+p3*exp(v3+sigma3^2/2);
% a2 = St*exp((r-delta)*136/365)
% a1-a2

% integrate from z, xi is the strike price points


fun = @(x) p1*lognpdf(x,v1,sigma1)+p2*lognpdf(x,v2,sigma2)+p3*lognpdf(x,v3,sigma3);



for i = 1:length(z)

    fun_call = @(x) (x-z(i)).*fun(x);
    fun_put = @(x) (z(i)-x).*fun(x);

%q = integral(fun,z,Inf,'ArrayValued',true)
    c(i) = exp(-r*dT)*integral(fun_call,z(i),Inf);
    p(i) = exp(-r*dT)*integral(fun_put, 0, z(i));
end


##################

function[x, fval] = pengbofeng

n = 41;
global K
global gamma
global lambda
K=linspace(800, 2400, n); % Starting guess for strike prices


St = 1920.24;
lambda = 100;
gamma = 30;

r = 0.003;
delta = 0.021;
dT = 136 / 365;

for i = 1:n,
    B0(i,:) = c_int(K(i), K, gamma);
end

B1 = exp(-r * dT) * B0;
B = [B1;

-B1;

zeros(n, n)];

A = [eye(n), -eye(n), zeros(n);

-eye(n), -eye(n), zeros(n);

zeros(n), -eye(n), zeros(n);

B, zeros(3 * n, n), kron(ones(3, 1), -eye(n))];

[c, p] = real_option_price(K);