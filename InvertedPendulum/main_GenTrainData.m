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
par.tf = 1;

[sys,par] = pendulum(par);

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);

dx0 = [0;0];
lbu = -200*ones(nu,1);
ubu = 200*ones(nu,1);
u0  = 0;

u_init = 0;
d_init = [2;pi];

lbx = [0,-5]';
ubx = [2*pi,5]';

par.N = 30;

par.nData = 10; % no. of data points along each dimension solved as NLP
par.nAug = 10; % no. of data points along each dimension solved using sensitivity update

par.ROC = 0;
[solver,par] = buildNLP(sys.f,par);

n_w_i = nx + par.N*(4*nx+nu);
sensitivity = 1;

x1 = linspace(lbx(1),ubx(1),par.nData);
x2 = linspace(lbx(2),ubx(2),par.nData);

k = 0;
for i =1:numel(x1)
    for j = 1:numel(x2)
        k = k+1;
        xi(:,k) = [x1(i);x2(j)];
    end
end

i = 0;
for sim_k = 1:par.nData^nx
    
    d_i = d_init;
    u_i = u_init;
    x_i = xi(:,sim_k);
    
    
    % ------ SOLVE OFFLINE OCP -------
    tic;
    sol = solver('x0',par.w0,'p',vertcat(x_i,u_init,d_init),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
    elapsednlp = toc;
    
    flag = solver.stats();
    exitflag =  flag.return_status;
    disp(['Data point nr. ' num2str(i+1)  ' - ' exitflag '. CPU time: ' num2str(elapsednlp) 's'])
    
    Primal = full(sol.x);
    Dual.lam_g = full(sol.lam_g);
    Dual.lam_x = full(sol.lam_x);
    Dual.lam_p = full(sol.lam_p);
    
    indAS = find(round(Dual.lam_x,6) >0);
    
    u1_opt = [Primal(nx+1:4*nx+nu:n_w_i);NaN];
    x1_opt = Primal([1,nu+4*nx+1:4*nx+nu:n_w_i]);
    x2_opt = Primal([2,nu+4*nx+2:4*nx+nu:n_w_i]);
    
    if flag.success
        i = i+1;
        GenData.u(:,i) = u1_opt(1);
        GenData.x(:,i) = vertcat(x1_opt(1),x2_opt(1),d_i);
        GenData.sol_t(i) = elapsednlp;
        
        GenData1.u(:,i) = u1_opt(1);
        GenData1.x(:,i) = vertcat(x1_opt(1),x2_opt(1),d_i);
        GenData1.sol_t(i) = elapsednlp;
        
        GenData0.u(:,i) = u1_opt(1);
        GenData0.x(:,i) = vertcat(x1_opt(1),x2_opt(1),d_i);
        GenData0.sol_t(i) = elapsednlp;
    end
    
    if sensitivity && flag.success
        
        % ------- Data Augmentation -------
        
        x1 = max(lbx(1),min(ubx(1),linspace(x_i(1)-0.3491,x_i(1)+0.3491,par.nAug)));
        x2 = max(lbx(2),min(ubx(2),linspace(x_i(2)-0.55,x_i(2)+0.55,par.nAug)));
        
        k = 0;
        for ij =1:numel(x1)
            for ji = 1:numel(x2)
                k = k+1;
                xip(:,k) = [x1(ij);x2(ji)];
            end
        end
        for j = 1:par.nAug*par.nAug
            
            p_init = vertcat(x_i,u_i,d_i);
            
            d_p = d_init;
            x_p = xip(:,j);
            
            p_final = vertcat(x_p,u_i,d_p);
            
            % ----------- full NLP ------------
            tic;
            sol1 = solver('x0',par.w0,'p',vertcat(x_p,u_i,d_p),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
            elapsednlp1 = toc;
            flag1 = solver.stats();
            
            Primal1 = full(sol1.x);
            Dual1.lam_g = full(sol1.lam_g);
            Dual1.lam_x = full(sol1.lam_x);
            
            u1_opt_p1 = [Primal1(nx+1:4*nx+nu:n_w_i);NaN];
            
            if flag1.success
                i = i+1;
                GenData1.u(:,i) = u1_opt_p1(1);
                GenData1.x(:,i) = vertcat(x_p,d_p);
                GenData1.sol_t(i) = elapsednlp1;
                
                GenData0.u(:,i) = NaN;
                GenData0.x(:,i) = NaN*vertcat(x1_opt(1),x2_opt(1),d_i);
%                 GenData0.sol_t(i) = NaN;
            end
            
            if flag1.success
                % ----------- Tangential PRedictor ------------
                if j == 1
                    [solLS,elapsed,H] = SolveLinSysOnline(Primal,Dual,vertcat(x_i,u_init,d_init),vertcat(x_p,u_i,d_p),par);
                    nw = numel(Primal);
                    ng = numel(Dual.lam_g);
                else
                    dp = (vertcat(x_p,u_i,d_p)-vertcat(x_i,u_init,d_init));
                    tic
                    Delta_s = H*dp;
                    elapsed = toc;
                    
                    solLS.dx = Delta_s(1:nw);
                    solLS.lam_g = Delta_s(nw+1:nw+ng);
                    solLS.lam_x = Delta_s(nw+ng+1:nw+ng+1);
                end
                w_opt_p = Primal + full(solLS.dx);
                lam_g_p = Dual.lam_g + full(solLS.lam_g);
                lam_x_p = Dual.lam_x + full(solLS.lam_x);
                
                disp(['Data point nr. ' num2str(i) '. CPU time: ' num2str(elapsed) 's'])
                
                u1_opt_p = [w_opt_p(nx+1:4*nx+nu:n_w_i);NaN];
                x1_opt_p = w_opt_p([1,nu+4*nx+1:4*nx+nu:n_w_i]);
                x2_opt_p = w_opt_p([2,nu+4*nx+2:4*nx+nu:n_w_i]);
                
                %                 indpAS = find(round(lam_x_p,6) >0);
                
                %                 if indAS == indpAS % Check if active set is same
                GenData.u(:,i) = u1_opt_p(1);
                GenData.x(:,i) = vertcat(x1_opt_p(1),x2_opt_p(1),d_p);
                GenData.sol_t(i) = elapsed;
                %                 end
            end
        end
    end
end

save('data/GenData_Aug','GenData')
save('data/GenData_full','GenData1')
save('data/GenData_sparse','GenData0')
