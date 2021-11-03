clc
clear
% uses CasADi v3.5.1
% www.casadi.org
import casadi.*

% Written by: Dinesh Krishnamoorthy, Nov 2021
%%
global nx nu nd
global lbx ubx dx0 lbu ubu u0
nx =2; nu = 1; nd = 2;

par.tf = 1;

[sys,par] = pendulum(par);

dx0 = [0;0];
lbu = -200*ones(nu,1);
ubu = 200*ones(nu,1);
u0  = 0;

lbx = [0,-5]';
ubx = [2*pi,5]';

d_val = [2;pi];

par.N = 30;
par.ROC = 0;
[solver,par] = buildNLP(sys.f,par);
n_w_i = nx + par.N*(4*nx+nu);

par.nIter = 40;

K = [-11,-7,35]; % initial policy which is linear

par.nAug = 5;
nr = 0;
nr2 = 0;

AugmentedFeedback = 1;

for rollout = 1:5
    disp(['Rollout: ' num2str(rollout)])
    switch rollout
        case 1
            xf = [3.1;0];
        case 2
            xf =[1;3.2];
        case 3
            xf = [5;-4];
        case 4
            xf = [3;4];
        case 5
            xf = [-2;4];
        case 6
            xf = [1;-3];
        case 7
            xf = [1.5;-0.5];
    end
    for run = 1:2
        if run == 1 % Current Policy Rollout
            for t = 1:par.nIter
                if rollout == 1
                    NMPC.u(t) = K(1:2)*xf + K(3);
                else
%                     NMPC.u(t) = predict(Approx_policy,xf');
NMPC.u(t) = sim(Approx_policy,xf);
                end
                NMPC.x(:,t) = xf;
                %------------------------ Plant simulation----------------------
                u_in = NMPC.u(t);
                Fk = sys.F('x0',xf,'p',vertcat(u_in,d_val));
                xf =  full(Fk.xf) ;
            end
        end
        
        % ===================================================================
        
        if run == 2 % Get expert feedback
            
            for t = 1:par.nIter
                
                
                tic;
                sol = solver('x0',par.w0,'p',vertcat(NMPC.x(:,t),0,d_val),...
                    'lbx',par.lbw,'ubx',par.ubw,...
                    'lbg',par.lbg,'ubg',par.ubg);
                elapsednlp = toc;
                
                flag = solver.stats();
                if ~flag.success
                    warning(['Expert says: ' flag.return_status])
                else
                    Primal = full(sol.x);
                    Dual.lam_g = full(sol.lam_g);
                    Dual.lam_x = full(sol.lam_x);
                    Dual.lam_p = full(sol.lam_p);
                    
                    u1_opt = [Primal(nx+1:4*nx+nu:n_w_i);NaN];
                    x1_opt = Primal([1,nu+4*nx+1:4*nx+nu:n_w_i]);
                    x2_opt = Primal([2,nu+4*nx+2:4*nx+nu:n_w_i]);
                    nr2 = nr2+1;
                    Expert.x(:,nr2) = NMPC.x(:,t);
                    Expert.u(nr2) = Primal(nx+1);
                    Expert.sol_t(nr2) = elapsednlp;
                end
                x_i = NMPC.x(:,t);
                u_i = 0;
                d_i = d_val;
                
                if t>1
                    dX = norm(x_i - NMPC.x(:,t-1));
                else
                    dX = 1;
                end
                if flag.success && dX >0.1% Augment additional samples from feedback
                   
                    % ------- Data Augmentation -------
                    
                    fr = 2*pi*rand(par.nAug*par.nAug,1);
                    r = 0.6*sqrt(rand(par.nAug*par.nAug,1));
                    x1 = x_i(1) + r.*cos(fr);
                    x2 = x_i(2) + r.*sin(fr);
                    
                    xip = [x1,x2]';
                    
                    for j = 1:par.nAug*par.nAug
                        % ----------- Tangential PRedictor ------------
                        nr = nr+1;
                        AugData.u(:,nr) = Expert.u(t);
                        AugData.x(:,nr) = Expert.x(:,t);
                        AugData.sol_t(nr) = Expert.sol_t(t);
                        nr = nr+1;
                        if j == 1
                            [solLS,elapsed,H] = SolveLinSysOnline(Primal,Dual,vertcat(x_i,u_i,d_i),vertcat(xip(:,j),u_i,d_i),par);
                            nw = numel(Primal);
                            ng = numel(Dual.lam_g);
                        else
                            dp = (vertcat(xip(:,j),u_i,d_i)-vertcat(x_i,u_i,d_i));
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
                        
                        %                         disp(['Data point nr. ' num2str(nr) '. CPU time ratio: ' num2str(Expert.sol_t(t)/elapsed)])
                        
                        u1_opt_p = [w_opt_p(nx+1:4*nx+nu:n_w_i);NaN];
                        x1_opt_p = w_opt_p([1,nu+4*nx+1:4*nx+nu:n_w_i]);
                        x2_opt_p = w_opt_p([2,nu+4*nx+2:4*nx+nu:n_w_i]);
                        
                        AugData.u(:,nr) = u1_opt_p(1);
                        AugData.x(:,nr) = vertcat(x1_opt_p(1),x2_opt_p(1));
                        AugData.sol_t(nr) = elapsed;
                        
                    end
                    
                end
            end
        end
    end
    
    %% Update Policy
    
    if AugmentedFeedback
%         Approx_policy = fitrgp(AugData.x',AugData.u','KernelFunction','ardsquaredexponential','Basis','none');
Approx_policy = newgrnn(AugData.x,AugData.u,0.25);

    else
%         Approx_policy = fitrgp(Expert.x',Expert.u','KernelFunction','ardsquaredexponential','Basis','none');
Approx_policy = newgrnn(Expert.x,Expert.u,0.25);
    end
    
    x1 = linspace(lbx(1),ubx(1),40);
    x2 = linspace(lbx(2),ubx(2),40);
    k = 0;
    for i =1:numel(x1)
        for j = 1:numel(x2)
            k = k+1;
            xi(:,k) = [x1(i);x2(j)];
        end
    end
    
%     [upred,~,uint] = predict(Approx_policy,xi');
% upred = Approx_policy(xi);
    upred = sim(Approx_policy,xi);
    %%
    figure(12)
    clf
    hold all
    plot3(Expert.x(1,:),Expert.x(2,:),Expert.u,'ro','linewidth',1.5)
    if AugmentedFeedback
    plot3(AugData.x(1,:),AugData.x(2,:),AugData.u,'k.')
    end
    plot3(xi(1,:),xi(2,:),upred,'.')
    hold all
    xlim([0,7])
    ylim([-5,5])
    grid on
    
    load('GenData_full.mat')
    plot3(GenData1.x(1,:),GenData1.x(2,:),GenData1.u,'.','markersize',1)
    
end
