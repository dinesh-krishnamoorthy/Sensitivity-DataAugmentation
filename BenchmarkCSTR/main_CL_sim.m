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

%% Train Approximate MPC

policy = 'Aug';

if strcmp(policy,'mpc')==0
    switch policy
        case 'full'
            load('data/GenData_full.mat')
            TrainData = GenData1;
        case 'sparse'
            load('data/GenData_sparse.mat')
            TrainData = GenData0;
        case 'Aug'
            load('data/GenData_Aug.mat')
            TrainData = GenData;
    end
    
    
    x = TrainData.x(1:2,:)';
    u = TrainData.u';
    
    nNeurons = 10;
    nLayers = 5;
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation (default).
     hiddenLayerSize = nNeurons.*ones(1,nLayers);
    
    net = fitnet(hiddenLayerSize,trainFcn);
    
    % Create a Fitting Network
%     for i = 1:nLayers
%         net.layers{i}.transferFcn = 'poslin';
%     end
    
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 75/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,tr] = train(net,x',u');
    
    % Test the Network
    u_opt = net(x');
    e = gsubtract(u,u_opt);
    performance = perform(net,u,u_opt);
    
    switch policy
        case 'full'
            genFunction(net,'ApproxMPC');
            genFunction(net,'ApproxMPC_full');
        case 'sparse'
            genFunction(net,'ApproxMPC');
            genFunction(net,'ApproxMPC_sparse');
        case 'Aug'
            genFunction(net,'ApproxMPC');
            genFunction(net,'ApproxMPC_Aug');
    end
    
end

%%

global nx nu nd
global lbx ubx dx0 lbu ubu u0
par.tf = 3;

[sys,par] = BenchmarkCSTR(par);

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);


lbx = [0.2632;0.6519] - [0.2;0.2];
ubx = [0.2632;0.6519] + [0.2;0.2];
dx0 = [0.98;0.39];
lbu = 0*ones(nu,1);
ubu = 2*ones(nu,1);
u0  = 1;

u_in = 0.1;
d_val = [0.2632;0.6519];
xf = [0.2;0.4];

if strcmp(policy,'mpc')
    par.N = 140;
    par.ROC = 0;
    [solver,par] = buildNLP(sys.f,par);
    n_w_i = nx + par.N*(4*nx+nu);
end

par.nIter = 140;

for sim_k = 1:par.nIter
    
    
     if strcmp(policy,'mpc')==0
        % Approximate MPC policy
        tic;
        NMPC.u(:,sim_k) = ApproxMPC(vertcat(xf));
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

switch policy
    case 'full'
        save('data/approx_nmpc_full','NMPC')
    case 'sparse'
        save('data/approx_nmpc_sparse','NMPC')
    case 'Aug'
        save('data/approx_nmpc_Aug','NMPC')
    case 'mpc'
        save('data/nmpc','NMPC')
end
%%
mpc = load('data/nmpc.mat');
full1 = load('data/approx_nmpc_full.mat');
sparse = load('data/approx_nmpc_sparse.mat');
Aug = load('data/approx_nmpc_Aug.mat');


load('data/GenData_sparse.mat')
load('data/GenData_Aug.mat')
load('data/GenData_full.mat')

figure(57)
clf
hold all
plot(GenData1.x(1,:),GenData1.x(2,:),'.','markersize',1,'color',[1,1,1]*0.5)
plot(GenData.x(1,:),GenData.x(2,:),'.','markersize',1,'color',[1,1,1]*0)
plot(GenData0.x(1,:),GenData0.x(2,:),'ro','markersize',3,'linewidth',1)
ylabel('Temperature $x_2$','Interpreter','latex')
xlabel('Concentration $x_1$','Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on

x1 = lbx(1):0.01:ubx(1);
x2 = lbx(2):0.01:ubx(2);
[X1,X2] = meshgrid(x1,x2);
for i = 1:numel(x1)
    for j = 1:numel(x2)
U(i,j) = ApproxMPC(vertcat(x1(i),x2(j)));
    end
end
figure(55)
hold all
plot3(TrainData.x(1,:),TrainData.x(2,:),TrainData.u,'k.')
surf(X1',X2',U)

figure(54)
hold all
plot(abs(GenData.u-GenData1.u))


time = [1:numel(mpc.NMPC.u)];
figure(56)
clf
hold all
stairs(time,mpc.NMPC.u,'b','linewidth',2)
stairs(time,sparse.NMPC.u,'linewidth',2,'color',[1,1,1]*0.5)
stairs(time,full1.NMPC.u,'linewidth',2,'color',[0,0.5,0])
stairs(time,Aug.NMPC.u,'r','linewidth',2)
ylabel('$u [kW]$','Interpreter','latex')
legend('$\pi_{mpc}(x)$','$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_1)$','$\pi_{approx}(x,\theta_2)$',...
    'Interpreter','latex')

xlabel('Time $ [h]$','Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on
grid on




