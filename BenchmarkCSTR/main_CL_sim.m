clc
clear

% uses CasADi v3.5.1
% www.casadi.org
% Writte by: Dinesh Krishnamoorthy

import casadi.*

FileName = mfilename('fullpath');
[directory,~,~] = fileparts(FileName);
[parent,~,~] = fileparts(directory);
addpath([directory '/data'])
addpath([directory '/model'])
addpath([parent '/functions'])

%%

global nx nu nd
global lbx ubx dx0 lbu ubu u0
par.tf = 3;

[sys,par] = BenchmarkCSTR(par);

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);


lbx = 0.*ones(nx,1);
ubx = 1*ones(nx,1);
dx0 = [0.98;0.39];
lbu = 0*ones(nu,1);
ubu = 2*ones(nu,1);
u0  = 1;

u_in = 0.1;
d_val = [0.2632;0.6519];
xf = [0.2;0.4];

ApproxPolicy = 0;

if ~ApproxPolicy
    par.N = 140;
    par.ROC = 0;
    [solver,par] = buildNLP(sys.f,par);
    n_w_i = nx + par.N*(4*nx+nu);
end

par.nIter = 140;

for sim_k = 1:par.nIter
    
    
    if ApproxPolicy
        % Approximate MPC
        tic;
        NMPC.u(:,sim_k) = max(0,min(2,ApproxMPC(vertcat(xf,d_val))));
        NMPC.sol_t(sim_k) = toc;
    else
        % Traditional MPC
        tic;
        sol = solver('x0',par.w0,'p',vertcat(xf,u_in,d_val),...
            'lbx',par.lbw,'ubx',par.ubw,...
            'lbg',par.lbg,'ubg',par.ubg);
        NMPC.sol_t(sim_k) = toc;
        
        flag = solver.stats();
        assert(flag.success, ['Error: ' flag.return_status])
        
        Primal = full(sol.x);
        
        u1_opt = [Primal(nx+1:4*nx+nu:n_w_i);NaN];
        x1_opt = Primal([1,nu+4*nx+1:4*nx+nu:n_w_i]);
        x2_opt = Primal([2,nu+4*nx+2:4*nx+nu:n_w_i]);
        
        NMPC.u(:,sim_k) = [Primal(nx+1)];
    end
    
    %------------------------ Plant simulation----------------------
    u_in = NMPC.u(:,sim_k);
    Fk = sys.F('x0',xf,'p',vertcat(u_in,d_val));
    xf =  full(Fk.xf) ;
    
    NMPC.J(sim_k) = full(Fk.qf);
    NMPC.x(:,sim_k) = xf;
    
end

if ApproxPolicy
    save('data/approx_nmpc','NMPC')
else
    save('data/nmpc','NMPC')
end





