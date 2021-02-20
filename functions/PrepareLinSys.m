function [phi,d_phi,c,H,Lpx,d_g_x,d_g_p,Lx_val]= PrepareLinSys(p_val,Primal,Dual,par)
import casadi.*

w = par.nlp.x;
p = par.nlp.p;
g = par.nlp.g;
J = par.nlp.f;

phi = Function('phi', {w,p}, {J},{'w','p'},{'J'});
c = Function('c', {w,p}, {g},{'w','p'},{'g'});

Lagr_func = J + Dual.lam_g'*g + Dual.lam_x'*w;

d_phi = phi.factory('d_phi',{'w','p'},{'jac:J:w'});

lagrangian = Function('lagrangian',{w,p},{Lagr_func},{'w','p'},{'Lagr_func'});
H = lagrangian.factory('H',{'w','p'},{'hess:Lagr_func:w:w'});

Lp = Function('Lp',{w,p},{jacobian(Lagr_func,p)},{'w','p'},{'Lp'});
Lx = Function('Lx',{w,p},{jacobian(Lagr_func,w)},{'w','p'},{'Lw'});
Lpx = Function('Lpx',{w,p},{jacobian(jacobian(Lagr_func,p),w)},{'w','p'},{'Lpw'});

d_g_x = c.factory('d_g_x',{'w','p'},{'jac:g:w'});
d_g_p = c.factory('d_g_p',{'w','p'},{'jac:g:p'});


phi = phi(Primal,p_val);
d_phi = d_phi(Primal,p_val);
c = c(Primal,p_val);
H = H(Primal,p_val);
Lp_val = Lp(Primal,p_val);
Lx_val = Lx(Primal,p_val);
Lpx = Lpx(Primal,p_val);
d_g_x = d_g_x(Primal,p_val);
d_g_p = d_g_p(Primal,p_val);

end



