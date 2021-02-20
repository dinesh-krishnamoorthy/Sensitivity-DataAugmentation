function [xf,exitflag] = solveODE(sys,d_val,u_in,opts)

% Function that computes the steady-state optimum
% Written by Dinesh Krishnamoorthy, Jul 2019, NTNU

import casadi.*

if nargin<8
    opts = struct('warn_initial_bounds',false, ...
        'print_time',false, ...
        'ipopt',struct('print_level',1)...
        );
end

lbx = 0.*ones(size(sys.x));
ubx = 1e5.*ones(size(sys.x));
dx0 = -1e5.*ones(size(sys.x));

assert(numel(sys.u)==numel(u_in))
assert(numel(sys.d)==numel(d_val))

w = {};
w0 = [];
lbw = [];
ubw = [];

g = {};
lbg = [];
ubg = [];

w = {w{:},sys.x,sys.u};
lbw = [lbw;lbx;u_in];
ubw = [ubw;ubx;u_in];
w0 = [w0;dx0;u_in];

J = 0; % Economic objective

g = {g{:},vertcat(sys.dx)};
lbg = [lbg;zeros(numel(sys.dx),1)];
ubg = [ubg;zeros(numel(sys.dx),1)];

if ~isempty(sys.nlcon)
    assert(numel(sys.nlcon)==numel(sys.lb))
    assert(numel(sys.nlcon)==numel(sys.ub))
    
    g = {g{:},sys.nlcon};
    lbg = [lbg;sys.lb];
    ubg = [ubg;sys.ub];
end

nlp = struct('x',vertcat(w{:}),'p',sys.d,'f',J,'g',vertcat(g{:}));
solver = nlpsol('solver','ipopt',nlp,opts);

sol = solver('x0',w0,'p',d_val,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg);
wf = full(sol.x);
xf = wf(1:numel(sys.x));


flag = solver.stats();
exitflag =  flag.return_status;


