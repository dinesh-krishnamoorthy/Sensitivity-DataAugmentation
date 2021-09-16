function [sys,par] = CartPole(par)

% Cart-Pole with damped pendulum
% Written by D. Krishnamoorthy, Apr 2020

import casadi.*

% states x
x1 = MX.sym('x1'); % position x
x2 = MX.sym('x2'); % velocity x_dot
x3 = MX.sym('x3'); % angle theta
x4 = MX.sym('x4'); % angular velocity theta_dot

% input u
u = MX.sym('u'); % Force F

x_sp = MX.sym('x_sp',4); 

m_p = 1;
m_c = 1;
l = 0.5;
k = 0; % 10;

dx1 = x2;
dx2 = 1/(m_c + m_p*sin(x3)^2)*(u + m_p*sin(x3)*(l*x4^2 + 9.81*cos(x3)) - k*x4*cos(x3)/l);
dx3 = x4;
dx4 = 1/(l*(m_c + m_p*sin(x3)^2))*(-u*cos(x3) - m_p*l*x4^2*sin(x3)*cos(x3) - (m_c+m_p)*9.81*sin(x3) + (m_c+m_p)*k*x3/(m_p*l));

diff = vertcat(dx1,dx2,dx3,dx4);
x_var = vertcat(x1,x2,x3,x4);
d_var = x_sp;
p_var = vertcat(u);

L = sum((x_var-x_sp).^2) + 0*u^2; % cart-pole at x_sp = [0,pi,0,0] 

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



