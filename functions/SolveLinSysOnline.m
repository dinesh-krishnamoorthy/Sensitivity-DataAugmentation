function [sol,elapsedqp] = SolveLinSysOnline(Primal,Dual,p_init,p_final,par)

import casadi.*
global lbx ubx
dp = (p_final-p_init);

[H,Lpx,d_g_x,d_g_p]= PrepareLinSys(Dual,par);


H = H(Primal,p_init);
Lpx = Lpx(Primal,p_init);
d_g_x = d_g_x(Primal,p_init);
d_g_p = d_g_p(Primal,p_init);

nw = numel(Primal);
ng = numel(Dual.lam_g);

V = diag(Dual.lam_x);
X = diag(min(Primal-par.lbw,par.ubw-Primal));

M = [sparse(full(H))    , sparse(full(d_g_x))' , -eye(nw);...
    sparse(full(d_g_x)) , zeros(ng,ng)         , zeros(ng,nw);...
    V                   , zeros(nw,ng)         , X ];

N= [sparse(full(Lpx))*dp;...
    sparse(full(d_g_p))*dp ;...
    zeros(nw,1)];

tic
Delta_s = sparse(M)\-sparse(N);
elapsedqp = toc;

sol.dx = Delta_s(1:nw);
sol.lam_g = Delta_s(nw+1:nw+ng);
sol.lam_x = Delta_s(nw+ng+1:nw+ng+1);

end

