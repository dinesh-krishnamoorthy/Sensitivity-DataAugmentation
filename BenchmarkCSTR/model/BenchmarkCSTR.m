function [sys,par] = BenchmarkCSTR(par)

% Benhcmark CSTR from Hicks and Ray, 1971, modified by Kameswaran and
% Biegler, 2006. 
% Written by D. Krishnamoorthy, Apr 2020

import casadi.*

% states x
x1 = MX.sym('x1'); % Concentration
x2 = MX.sym('x2'); % Temperature

% input u
u = MX.sym('u'); % Energy flux from heating system

x_sp = MX.sym('x_sp',2); % Temperature

tau = 20;
M = 5;
xf = 0.3947;
xc = 0.3816;
a = 0.117;
k = 300;

dx1 = (1/tau)*(1-x1) - k*x1*exp(-M/x2);
dx2 = (1/tau)*(xf-x2) + k*x1*exp(-M/x2) - a*u*(x2-xc);

diff = vertcat(dx1,dx2);
x_var = vertcat(x1,x2);
d_var = x_sp;
p_var = vertcat(u);

L = sum(([x1;x2]-x_sp).^2) + 1e-4*u^2; % maintain desired temperature + min Uh heating costs? 

sys.f = Function('f',{x_var,p_var,d_var},{diff,L},{'x','p','d'},{'xdot','qj'});

ode = struct('x',x_var,'p',vertcat(p_var,d_var),'ode',diff,'quad',L); 

% create CVODES integrator
sys.F = integrator('F','cvodes',ode,struct('tf',par.tf));

sys.x = x_var;
sys.u = p_var;
sys.d = d_var;
sys.dx = diff;
sys.L = L;
sys.nlcon = [];



