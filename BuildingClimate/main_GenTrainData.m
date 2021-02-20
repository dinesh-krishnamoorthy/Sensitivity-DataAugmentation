
import casadi.*

clear
clc

global nx nu nd
global lbx ubx dx0 lbu ubu u0
par.tf = 1/60;

% parameters
par.Cs = 0.0549; % Heat capacities
par.Ci = 0.0928;
par.Ce = 3.32;
par.Ch = 0.889;

par.Ris = 1.89;
par.Rih = 0.146;
par.Rie = 0.897;
par.Rea = 4.38;
par.Ria = 2.5;

par.Aw = 5.75;
par.Ae = 3.87;

[sys,par] = BuildingModel(par);

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);


lbx = 12.*ones(nx,1);
ubx = 40*ones(nx,1);
dx0 = 20*ones(nx,1);
lbu = 0*ones(nu,1);
ubu = 40*ones(nu,1);
u0  = 0;

u_init = 2;
d_init = [0.05,1,22]';
x_init = solveODE(sys,d_init,u_init);

par.N = 3/par.tf;

par.nData = 30;
sensitivity = 1;
par.ROC = 0.1;
[solver,par] = buildNLP(sys.f,par);

plotData = 1;
i = 1;

n_w_i = nx + par.N*(4*nx+nu);

for sim_k = 1:par.nData
    
    x_init = lbx + (ubx-lbx).*rand(nx,1);
    d_init = [0;-5;18] + ([0.2;20;7]).*rand(nd,1);
    u_init = lbu + (ubu-lbu)*rand(nu,1);
    d_i = d_init;
    u_i = u_init;
    x_i = x_init;
    
    tic;
    sol = solver('x0',par.w0,'p',vertcat(x_init,u_init,d_init),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
    par.elapsednlp = toc;
    
    flag = solver.stats();
    exitflag =  flag.return_status;
    disp(['Data point nr. ' num2str(i)  ' - ' exitflag '. CPU time: ' num2str(par.elapsednlp) 's'])
    
    Primal = full(sol.x);
    Dual.lam_g = full(sol.lam_g);
    Dual.lam_x = full(sol.lam_x);
    
    u1_opt = [Primal(nx+1:4*nx+nu:n_w_i);NaN];
    
    if flag.success
        GenData.u(:,i) = u1_opt(1);
        GenData.x(:,i) = vertcat(x_init,u_init,d_i);
    end
    GenData.sol_t(i) = par.elapsednlp;
    i = i+1;
    
    if sensitivity && flag.success
        for j = 1:40
            
            p_init = vertcat(x_i,u_i,d_i);
            
            u_p = u_i.*0.8 + (u_i.*1.4 - u_i.*0.6).*rand(nu,1);
            x_p = x_i.*0.8 + (x_i.*1.4 - x_i.*0.6).*rand(nx,1);
            d_p = d_i.*0.8 + (d_i.*1.4 - d_i.*0.6).*rand(nd,1);
            p_final = vertcat(x_p,u_p,d_p);
            
            [solLS,elapsedqp] = SolveLinSysOnline(Primal,Dual,p_init,p_final,par);
            w_opt_p = Primal + full(solLS.dx);
            disp(['Data point nr. ' num2str(i) '. CPU time: ' num2str(elapsedqp) 's'])
            
            u1_opt_p = max(0,[w_opt_p(nx+1:4*nx+nu:n_w_i);NaN]);
            
            GenData.u(:,i) = u1_opt_p(1);
            GenData.x(:,i) = vertcat(x_p,u_p,d_i);
            
            GenData.sol_t(i) = elapsedqp;
            
            i = i+1;
        end
    end
end


save('GenData','GenData')

%% Train NN

load('GenData.mat')
x = GenData.x';
u = GenData.u';

nNeurons = 10;
nLayers = 3;
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation (default).
hiddenLayerSize = nNeurons.*ones(1,nLayers);


net = fitnet(hiddenLayerSize,trainFcn);

% Create a Fitting Network
for i = 1:nLayers
    net.layers{i}.transferFcn = 'poslin';
end


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x',u');

% Test the Network
u_opt = net(x');
e = gsubtract(u,u_opt);
performance = perform(net,u,u_opt);

genFunction(net,'ApproxMPC2');

