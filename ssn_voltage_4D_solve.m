% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)    

%% Solve SSN ODE (without noise term) 
% open ReLU.m and ssn_voltage_4Dode.m function files

% Time vector
dt = 0.003;
T0 = 0;
Tf = 0.5;
tspan = [T0:dt:Tf];

%u_0 = [-80; -60; -60; -60]
u_0 = ones(4,1);           % set initial condition to 1 to calculate trajectory


%% Solve using ode45
%use ode45 to numerical solve eq using Runge-Kutta 4th/5th order
[tout, u] = ode45(@ssn_voltage_4Dode, tspan, u_0);

% like uniform step sizes: interpolate the result
ui = interp1(tout,u,tspan);

x = u(1,:);
y = u(2,:);

figure(1);
%subplot(2,1,1)
plot(tout, u, 'Linewidth', 1.5)
ylabel("voltage - ode45 response")
%legend("E", "P", "V", "S")
%subplot(2,1,2)
%plot(tspan, ui, '-.', 'Linewidth', 1.5)
xlabel("time")
%ylabel("Interpolation response")
title("Time traces of response")
legend("E", "P", "V", "S")