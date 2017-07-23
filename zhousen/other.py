options = optimset('Display','iter','PlotFcns',@optimplotfval);
fun = @(x)100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
x0 = [-1.2,1];
[x,fval,exitflag,output] = fminsearch(fun,x0,options)



import scipy.optimize



banana = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
xopt = scipy.optimize.fmin(func=banana, x0=[-1.2,1])



import scipy.optimize as sc_opt

opts = {'maxiter': 100,
        'ftol': 10^-6,
        'iprint': 2,
        'disp': True}




sc_opt.minimize(log_barrier_risk_parity_obj_fun, current_weight, method='L-BFGS-B',
                                           jac=log_barrier_risk_parity_gradient, bounds=log_rp_bnds,
                                           options=log_barrier_risk_parity_opts)