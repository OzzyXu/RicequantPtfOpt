



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