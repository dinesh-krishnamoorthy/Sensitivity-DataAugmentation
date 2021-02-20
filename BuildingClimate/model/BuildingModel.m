function [sys,par] = BuildingModel(par)
import casadi.*

% Building climate control model taken from Bacher and Madsen (2011)
% Parameters taken from Table 3 (Bacher and Madsen, 2011)
%
% Written by Dinesh Krishnamoorthy, Dec 2019

% states x
Ts = MX.sym('Ts'); % Sensor temperature
Ti = MX.sym('Ti'); % Interior temperature
Th = MX.sym('Tc'); % Heater temperature
Te = MX.sym('Te'); % Tempreature of the building envelop

% disturbaces d
Ta = MX.sym('Ta'); % Ambient Tempreature 
Us = MX.sym('Us'); % Energy flux from solar radiation 
Td = MX.sym('Td'); % Desired temperature

% Ria  =MX.sym('Ria'); % thermal resistance between interior and ambient 

% input u
Uh = MX.sym('Uh'); % Energy flux from heating system

dTs =  1/(par.Ris*par.Cs)*(Ti-Ts);
dTi =  1/(par.Ris*par.Ci)*(Ts-Ti)  + ...
        1/(par.Rih*par.Ci)*(Th-Ti) + 1/(par.Rie*par.Ci)*(Te-Ti) +...
        1/(par.Ria*par.Ci)*(Ta-Ti) + 1/par.Ci*par.Aw*Us;
%dTm = 1/(Rim*Cm)*(Ti-Tm);
dTh = 1/(par.Rih*par.Ch)*(Ti-Th) + 1/par.Ch*Uh;
dTe = 1/(par.Rie*par.Ce)*(Ti-Te) + 1/(par.Rea*par.Ce)*(Ta-Te) ...
    + 1/par.Ce*(par.Ae*Us);

diff = vertcat(dTs,dTi,dTh,dTe);
x_var = vertcat(Ts,Ti,Th,Te);
d_var = vertcat(Us,Ta,Td);
p_var = vertcat(Uh);

L = (Ti-Td)^2; % maintain desired temperature + min Uh heating costs? 

sys.f = Function('f',{x_var,p_var,d_var},{diff,L},{'x','p','d'},{'xdot','qj'});

ode = struct('x',x_var,'p',vertcat(p_var,d_var),'ode',diff,'quad',L); 

% create CVODES integrator
sys.F = integrator('F','cvodes',ode,struct('tf',par.tf));

sys.x = x_var;
sys.u = p_var;
sys.d = d_var;
sys.dx = diff;
sys.L = L;
sys.nlcon = [];



