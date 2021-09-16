function [sys,par] = pendulum(par)

import casadi.*
m= 1;
l=2;

x = MX.sym('x',2);
u = MX.sym('u',1);
d = MX.sym('d',2); % damping factor b =2 and sp = pi

dx1 = x(2);
dx2 = (u - d(1)*x(2) - m*9.81*l*sin(x(1)))/(m*l^2);

dx = vertcat(dx1,dx2);

L = (x(1)-d(2)).^2 + (x(2)).^2;

% create CVODES integrator
ode = struct('x',x,'p',vertcat(u,d),'ode',dx,'quad',L);
opts = struct('tf',par.tf);
F = integrator('F','cvodes',ode,opts);

f = Function('f',{x,u,d},{dx,L},{'x','p','d'},{'xdot','qj'});

sys = struct('x',x,'u',u,'d',d,'dx',dx,'L',L,'f',f,'F',F);
