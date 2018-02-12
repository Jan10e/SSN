% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate ssn ode 4D network model, with rate as dynamic variable and OUP noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


%% Parameters 
% as in Kuchibotla
k = 0.01; %0.01; scaling constant 
n = 2.2;  %2.2

% Connectivity Matrix W as in Kuchibotla, Miller & Froemke
w_EE = .017; w_EP = -.956; w_EV = -.045; w_ES = -.512;
w_PE = .8535; w_PP = -.99; w_PV = -.09; w_PS = -.307;
w_VE = 2.104; w_VP = -.184; w_VV = 0; w_VS = -.734;
w_SE = 1.285; w_SP = 0; w_SV = -.14; w_SS = 0;

W = [w_EE w_EP w_EV w_ES;
    w_PE w_PP w_PV w_PS;
    w_VE w_VP w_VV w_VS;
    w_SE w_SP w_SV w_SS];


% Membrane time constant 
tau_E = 20/1000; %ms; 20ms for E
tau_P = 10/1000; %ms; 10ms for all PV
tau_V = 10/1000; %ms; 10ms for all VIP
tau_S = 10/1000; %ms; 10ms for all SOM
tau = [tau_E; tau_P; tau_V; tau_S];


% Input
%Hh = zeros(4,1);              %input; no input = 0; somewhat larger input = 2; large input = 15
Hh = [0; 2; 15; 0];             %E, P, V, S

%% Functions

% ODE rate with t and u only
ode_rate2 = @(t, u)  (-u + k.*ReLU(W *u + Hh).^n)./tau;



%% Solve 4D ODE

Uu_0 = ones(4,1);           % set initial condition to 1 to calculate trajectory
tspan = (0:0.003:5); 

% use ode45 to numerical solve eq using Runge-Kutta 4th/5th order
[t, u] = ode45(ode_rate2, tspan, Uu_0);

figure;
plot(t, u, 'Linewidth', 2)
ylabel("rate")
xlabel("time")
legend("E", "P", "V", "S")
    

