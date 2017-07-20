import numpy as np
from scipy.optimize import minimize
from scipy.optimize import *
from pandas import *

stocks = index_components("000300.XSHG")
price = get_price(stocks,start_date = '2016-01-01',end_date = '2017-01-01',fields = 'close')
price = price.T.dropna(how = 'any').T

corr = price.cov()

def func(x,corr = corr):
    return np.dot(np.dot(x,corr),x)

q = [{'type':'eq','fun':lambda x:x.sum()-1}]
q.append({'type':'ineq','fun':lambda x:x[0]})
q.append({'type':'ineq','fun':lambda x:x[1]})
q.append({'type':'ineq','fun':lambda x:x[2]})
q.append({'type':'ineq','fun':lambda x:x[3]})
q.append({'type':'ineq','fun':lambda x:x[4]})
q.append({'type':'ineq','fun':lambda x:x[5]})
q.append({'type':'ineq','fun':lambda x:x[6]})
q.append({'type':'ineq','fun':lambda x:x[7]})
q.append({'type':'ineq','fun':lambda x:x[8]})
q.append({'type':'ineq','fun':lambda x:x[9]})
for i in range(10, 282):
    def test(x):
        return x[i]
    q.append({'type':'ineq','fun':test})
cons_ = tuple(q)

x0 = np.ones(282)/282
res = minimize(func, x0, constraints=cons_, method='SLSQP', options={'disp': True, "iprint":2,"maxiter":1000})
x_ = res.x

x_


x1 = np.zeros(282)

x1[20] = 100
print(x1[:22])
len(cons_)
cons_[21]["fun"](x1)




q = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
q.append({'type': 'ineq', 'fun': lambda x: x[0]})
q.append({'type': 'ineq', 'fun': lambda x: x[1]})




zs([1,2,3])

