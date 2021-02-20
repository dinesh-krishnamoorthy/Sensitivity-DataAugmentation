load('data/GenData.mat');
d_init = [0.2632;0.6519];
par.nData = 20;
lbx = d_init - [0.2;0.2];
ubx = d_init + [0.2;0.2];

x1 = linspace(lbx(1),ubx(1),par.nData);
x2 = linspace(lbx(2),ubx(2),par.nData);

k = 0;
for i =1:numel(x1)
    for j = 1:numel(x2)
        k = k+1;
        xi(:,k) = [x1(i);x2(j)];
    end
end

t1 = GenData.sol_t(1:26:end);
ind = find(GenData.sol_t<min(t1));
ind2 = find(GenData.sol_t>=min(t1));

t2 = GenData.sol_t(ind);

CPU1.min = min(t1);
CPU1.avg = mean(t1);
CPU1.max = max(t1);

CPU2.min = min(t2);
CPU2.avg = mean(t2);
CPU2.max = max(t2);

figure(45)
clf
hold all
plot(GenData.x(1,ind),GenData.x(2,ind),'k.','markersize',6)
% plot(xi(1,:),xi(2,:),'ro','linewidth',1)
plot(GenData.x(1,ind2),GenData.x(2,ind2),'ro','linewidth',1)
xlim([lbx(1),ubx(1)])
ylim([lbx(2),ubx(2)])
yticks([lbx(2):0.05:ubx(2)])
xticks([lbx(1):0.05:ubx(1)])
axs = gca;
axs.TickLabelInterpreter = 'latex';
axs.FontSize = 14;
xlabel('concentration','Interpreter','latex')
ylabel('Temperature','Interpreter','latex')

%%

NN = load('data/approx_nmpc.mat');
MPC = load('data/nmpc.mat');

figure(46)
clf
hold all
stairs(MPC.NMPC.u,'b','linewidth',2.5)
stairs(NN.NMPC.u,'r','linewidth',2.5)
axs = gca;
axs.TickLabelInterpreter = 'latex';
axs.FontSize = 14;
box on
grid on
legend('$\pi^*_{mpc}(x)$','$\pi^*_{approx}(x)$','Interpreter','latex')
ylabel('$u$','Interpreter','latex')
xlabel('time ','Interpreter','latex')







