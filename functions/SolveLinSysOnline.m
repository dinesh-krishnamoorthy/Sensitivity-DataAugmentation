function [sol,elapsedqp] = SolveLinSysOnline(Primal,Dual,p_init,p_final,par)

import casadi.*

dp = (p_final-p_init);

[phi,d_phi,c,H,Lpx,d_g_x,d_g_p,Lx_val]= PrepareLinSys(p_init,Primal,Dual,par);

nw = numel(Primal);
ng = numel(Dual.lam_g);

V = diag(Dual.lam_x);
X = diag(Primal);


M = [sparse(full(H))    , sparse(full(d_g_x))' , -eye(nw);...
    sparse(full(d_g_x)) , zeros(ng,ng)         , zeros(ng,nw);...
    V                   , zeros(nw,ng)         , X ];

N= [sparse(full(Lpx))'*dp;...
    sparse(full(d_g_p))*dp ;...
    zeros(nw,1)];

tic
Delta_s = M\-N;
elapsedqp = toc;

sol.dx = Delta_s(1:nw);
sol.lam_g = Delta_s(nw+1:nw+ng);
sol.lam_x = Delta_s(nw+ng+1:nw+ng+nw);

end

