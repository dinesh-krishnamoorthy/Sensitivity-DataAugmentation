function [H,Lpx,d_g_x,d_g_p]= PrepareLinSys(Dual,par)
import casadi.*

w = par.nlp.x;
p = par.nlp.p;
g = par.nlp.g;
J = par.nlp.f;


Lagr_func = J + Dual.lam_g'*g + Dual.lam_x'*w;

Lpx = Function('Lpx',{w,p},{jacobian(jacobian(Lagr_func,w),p)},{'w','p'},{'Lpw'});
H = Function('H',{w,p},{jacobian(jacobian(Lagr_func,w),w)},{'w','p'},{'H'});
d_g_x = Function('d_g_x',{w,p},{jacobian(g,w)},{'w','p'},{'d_g_x'}); %c.factory('d_g_x',{'w','p'},{'jac:g:w'});
d_g_p = Function('d_g_p',{w,p},{jacobian(g,p)},{'w','p'},{'d_g_p'}); %c.factory('d_g_p',{'w','p'},{'jac:g:p'});


end



