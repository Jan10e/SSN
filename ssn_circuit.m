% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)

%% Parameters
% % Connectivity Matrix W
% w_EE = 1.25;
% w_EI = -0.65;
% w_IE = 1.2;
% w_II = -0.5;
% W = [w_EE w_EI; w_IE w_II];
% %w = [w_EE + w_IE ; w_II + w_EI];
% 
% % Membrane time constant
% tau_E = 20; %ms; membrane time constant (20ms for E)
% tau_I = 10; %ms; membrane time constant (10ms for I)
% tau = [(tau_E/100); (tau_I/100)];
% 
% % Input
% h = [0; 0]; %mV; no input = 0; somewhat larger input = 2; large input = 15

% % Parameters
% V_rest = -70;
% k = 0.3; %scaling constant 
% n = 2;

% Variables
%u = -70:0.1:80; %Vm cell i
u_0 = [-80; 60]; %-80 for E, 60 for I
%tspan = [0 5];
tspan =(0:0.03:5); % 30ms time bin


%% SSN
[t, u] = ode45(@ssn_ode, tspan, u_0);
plot(t, u)

% c =1; %constant, this should be the ReLU 
% 
% % Create symbolic funtion u(t) (V(t) in Hennequin)
% syms u(t)
% ode = diff(u, t) == ((-u +V_rest) + w * (k * (c).^n) + h)./tau;
% cond = u(0) == -70:0.1:80;
% ySol(t) = dsolve(ode, cond);


