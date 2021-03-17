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

dx0 = [0.98;0.39];
lbu = 0*ones(nu,1);
ubu = 2*ones(nu,1);
u0  = 0;

u_init = 1;
d_init = [0.2632;0.6519];

lbx = d_init - [0.2;0.2];
ubx = d_init + [0.2;0.2];

par.N = 140;

par.nData = 10; % no. of data points along each dimension solved as NLP
par.nAug = 20; % no. of data points along each dimension solved using sensitivity update

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
    
    indNLP = find((Primal(1:40)-par.lbw(1:40))<0);
    indNLP = vertcat(indNLP,find((par.ubw(1:40)-Primal(1:40))<0));
    
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
        
        x1 = max(lbx(1),min(ubx(1),linspace(x_i(1)-0.0105*2,x_i(1)+0.0105*2,par.nAug)));
        x2 = max(lbx(2),min(ubx(2),linspace(x_i(2)-0.0105*2,x_i(2)+0.0105*2,par.nAug)));
        
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
                GenData0.sol_t(i) = NaN;
            end
            
            
            if flag1.success
                % ----------- Tangential PRedictor ------------
                
                
                [solLS,elapsed] = SolveLinSysOnline(Primal,Dual,vertcat(x_i,u_init,d_init),vertcat(x_p,u_i,d_p),par);
                w_opt_p = Primal + full(solLS.dx);
                lam_g_p = Dual.lam_g + full(solLS.lam_g);
                lam_x_p = Dual.lam_x + full(solLS.lam_x);
                
                disp(['Data point nr. ' num2str(i) '. CPU time: ' num2str(elapsed) 's'])
                
                u1_opt_p = max(0,min(2,[w_opt_p(nx+1:4*nx+nu:n_w_i);NaN]));
                x1_opt_p = w_opt_p([1,nu+4*nx+1:4*nx+nu:n_w_i]);   
                x2_opt_p = w_opt_p([2,nu+4*nx+2:4*nx+nu:n_w_i]);
                
                indpNLP = find((w_opt_p(1:40)-par.lbw(1:40))<0);
                indpNLP = vertcat(indpNLP,find((par.ubw(1:40)-w_opt_p(1:40))<0));
                
                indpAS = find(round(lam_x_p,6) >0);
                
                %              if all(u1_opt_p)>lbu && all(u1_opt_p)<ubu && all(all([x1_opt_p,x2_opt_p]>lbx')) && all(all([x1_opt_p,x2_opt_p]<ubx'))
                
                GenData.u(:,i) = u1_opt_p(1);
                GenData.x(:,i) = vertcat(x1_opt_p(1),x2_opt_p(1),d_p);
                GenData.sol_t(i) = elapsed;

            end
        end
    end
end

save('data/GenDataA','GenData')

% %% Train DNN
%
% load('data/GenData_activeSet.mat')
% x = GenData.x';
% u = GenData.u';
%
% nNeurons = 10;
% nLayers = 3;
% trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation (default).
% hiddenLayerSize = nNeurons.*ones(1,nLayers);
%
% % Create a Fitting Network
% net = fitnet(hiddenLayerSize,trainFcn);
%
% % Setup Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
%
% % Train the Network
% [net,tr] = train(net,x',u');
%
% % Test the Network
% u_opt = net(x');
% e = gsubtract(u,u_opt);
% performance = perform(net,u,u_opt);
%
% % Generate Approximate MPC policy function
% genFunction(net,'ApproxMPC_activeSet');
