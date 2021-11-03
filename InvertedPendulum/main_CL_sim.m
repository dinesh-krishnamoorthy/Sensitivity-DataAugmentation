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

policy = 'mpc';
train = 0;
if train
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
end

%%

global nx nu nd
global lbx ubx dx0 lbu ubu u0
par.tf = 1;

[sys,par] = pendulum(par);
nx =2; nu = 1; nd = 2;
dx0 = [0;0];
lbu = -200*ones(nu,1);
ubu = 200*ones(nu,1);
u0  = 0;

lbx = [0,-5]';
ubx = [2*pi,5]';

u_in = 0;
d_val = [2;pi];
xf = [0;0];

if strcmp(policy,'mpc')
    par.N = 30;
    par.ROC = 0;
    [solver,par] = buildNLP(sys.f,par);
    n_w_i = nx + par.N*(4*nx+nu);
end

par.nIter = 140;

for sim_k = 1:par.nIter
    
    
    if strcmp(policy,'mpc')==0
        % Approximate MPC policy
        switch policy
            case 'full'
                tic;
                NMPC.u(:,sim_k) = ApproxMPC_full(vertcat(xf));
                NMPC.sol_t(sim_k) = toc;
            case 'sparse'
                tic;
                NMPC.u(:,sim_k) = ApproxMPC_sparse(vertcat(xf));
                NMPC.sol_t(sim_k) = toc;
            case 'Aug'
                tic;
                NMPC.u(:,sim_k) = ApproxMPC_Aug(vertcat(xf));
                NMPC.sol_t(sim_k) = toc;
        end
        
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
    NMPC.x(:,sim_k) = xf;
    %------------------------ Plant simulation----------------------
    u_in = NMPC.u(:,sim_k);
    Fk = sys.F('x0',xf,'p',vertcat(u_in,d_val));
    xf =  full(Fk.xf) ;
    
    NMPC.J(sim_k) = full(Fk.qf);
    
    
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


figure(50)
hold all
plot(mpc.NMPC.x(1,:),mpc.NMPC.x(2,:),'b','linewidth',2)
plot(sparse.NMPC.x(1,:),sparse.NMPC.x(2,:),'linewidth',2,'color',[1,1,1]*0.75)
plot(full1.NMPC.x(1,:),full1.NMPC.x(2,:),'linewidth',2,'color',[0,0.5,0])
plot(Aug.NMPC.x(1,:),Aug.NMPC.x(2,:),'r','linewidth',2)
ylabel('$ \theta [rad]$','Interpreter','latex')
legend('$\pi^*(x)$','$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_2)$','$\pi_{approx}(x,\theta_1)$',...
    'Interpreter','latex')

xlabel('Time $ [s]$','Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on
grid on



return
%%
load('data/GenData_sparse.mat')
load('data/GenData_Aug.mat')
load('data/GenData_full.mat')
time = 1:140;
figure(57)
clf
subplot(131)
hold all
plot3(GenData1.x(1,:),GenData1.x(2,:),GenData1.u,'.','markersize',1,'color',[1,1,1]*0.5)
plot3(GenData.x(1,:),GenData.x(2,:),GenData.u,'.','markersize',3,'color',[1,1,1]*0)
plot3(GenData0.x(1,:),GenData0.x(2,:),GenData0.u,'ro','markersize',6,'linewidth',1)
ylabel(' $\dot\theta$ [rad/s]','Interpreter','latex')
xlabel('$\theta$ [rad]','Interpreter','latex')
zlabel('$\mathbf{s}(x)$ ','Interpreter','latex')
legend('Augmented $\mathcal{D}^+$','Dense $\mathcal{D}^{++}$','Sparse $\mathcal{D}^0$',...
    'Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on

subplot(132)
hold all
plot3(GenData1.x(1,:),GenData1.x(2,:),GenData1.u,'.','markersize',1,'color',[1,1,1]*0.5)
plot3(GenData.x(1,:),GenData.x(2,:),GenData.u,'.','markersize',3,'color',[1,1,1]*0)
plot3(GenData0.x(1,:),GenData0.x(2,:),GenData0.u,'ro','markersize',6,'linewidth',1)
ylabel(' $\dot\theta$ [rad/s]','Interpreter','latex')
xlabel('$\theta$ [rad]','Interpreter','latex')
zlabel('$\mathbf{s}(x)$ ','Interpreter','latex')
legend('Augmented $\mathcal{D}^+$','Dense $\mathcal{D}^{++}$','Sparse $\mathcal{D}^0$',...
    'Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on


subplot(133)
hold all
plot3(GenData1.x(1,:),GenData1.x(2,:),abs(GenData1.u-GenData.u),'.','markersize',1)
zlim([0,20])
ylabel(' $\dot\theta$ [rad/s]','Interpreter','latex')
xlabel('$\theta$ [rad]','Interpreter','latex')
zlabel('$ \|\mathbf{s}^*(x) - \hat{\mathbf{s}}(x)\| $','Interpreter','latex')
legend('$|\mathcal{D}^+-\mathcal{D}^{++}|$',...
    'Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on
%%
% x1 = lbx(1):0.01:ubx(1);
% x2 = lbx(2):0.01:ubx(2);
% [X1,X2] = meshgrid(x1,x2);
% for i = 1:numel(x1)
%     for j = 1:numel(x2)
% U(i,j) = ApproxMPC(vertcat(x1(i),x2(j)));
%     end
% end
% figure(55)
% hold all
% plot3(TrainData.x(1,:),TrainData.x(2,:),TrainData.u,'k.')
% surf(X1',X2',U)

% figure(54)
% hold all
% plot(abs(GenData.u-GenData1.u))
% time = [1:numel(mpc.NMPC.u)];
figure(56)
clf
subplot(121)
hold all
stairs(time,mpc.NMPC.u,'b','linewidth',2)
stairs(time,sparse.NMPC.u,'linewidth',2,'color',[1,1,1]*0.75)
stairs(time,full1.NMPC.u,'linewidth',2,'color',[0,0.5,0])
stairs(time,Aug.NMPC.u,'r','linewidth',2)
ylabel('Torque $ [Nm]$','Interpreter','latex')
legend('$\pi^*(x)$','$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_2)$','$\pi_{approx}(x,\theta_1)$',...
    'Interpreter','latex')
xlim([0,60])

xlabel('Time $ [s]$','Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on
grid on

subplot(122)
hold all
plot(time,mpc.NMPC.x(1,:),'b','linewidth',2)
plot(time,sparse.NMPC.x(1,:),'linewidth',2,'color',[1,1,1]*0.75)
plot(time,full1.NMPC.x(1,:),'linewidth',2,'color',[0,0.5,0])
plot(time,Aug.NMPC.x(1,:),'r','linewidth',2)
ylabel('$ \theta [rad]$','Interpreter','latex')
legend('$\pi^*(x)$','$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_2)$','$\pi_{approx}(x,\theta_1)$',...
    'Interpreter','latex')
yticks([0,1,2,pi,4]);
% ylim([0,220])
xlabel('Time $ [s]$','Interpreter','latex')
axs = gca;
axs.FontSize = 14;
axs.TickLabelInterpreter = 'latex';
box on
grid on
xlim([0,60])


%% 
GenData0.mean_sol_t = mean(GenData0.sol_t);

GenData1.mean_sol_t = mean(GenData1.sol_t);
GenData1.mean_sol_t/GenData.mean_sol_t

GenData0.cum_sol_t = sum(GenData0.sol_t)
GenData.cum_sol_t = sum(GenData.sol_t)
GenData1.cum_sol_t = sum(GenData1.sol_t)

ind = find(GenData0.sol_t ==0);
ind1 = find(GenData0.sol_t >0);
GenData1.mean_sol_t1 = mean(GenData1.sol_t(ind));
GenData.mean_sol_t1 = mean(GenData.sol_t(ind));
GenData0.mean_sol_t = mean(GenData0.sol_t(ind1));
