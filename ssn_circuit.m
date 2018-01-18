% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)

%% Parameters
u_0 = [-80; 60]; %-80 for E, 60 for I
%u_0 = ones(2,1);

tspan = [0 5];
%tspan =(0:0.03:5); % 30ms time bin

%% SSN ODE
[t, u] = ode45(@ssn_ode, tspan, u_0);

figure;
plot(t, u)

%% To Do
% add input noise \eta > OU process (is ode45)

