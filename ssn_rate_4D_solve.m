% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate ssn ode 4D network model, with rate as dynamic variable and OUP noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


%% Parameters 
% as in Kuchibotla
param.k = 0.01; %0.01; scaling constant 
param.n = 2.2;  %2.2

% Connectivity Matrix W as in Kuchibotla, Miller & Froemke
param.w_EE = .017; param.w_EP = -.956; param.w_EV = -.045; param.w_ES = -.512;
param.w_PE = .8535; param.w_PP = -.99; param.w_PV = -.09; param.w_PS = -.307;
param.w_VE = 2.104; param.w_VP = -.184; param.w_VV = 0; param.w_VS = -.734;
param.w_SE = 1.285; param.w_SP = 0; param.w_SV = -.14; param.w_SS = 0;

param.W = [param.w_EE param.w_EP param.w_EV param.w_ES;
    param.w_PE param.w_PP param.w_PV param.w_PS;
    param.w_VE param.w_VP param.w_VV param.w_VS;
    param.w_SE param.w_SP param.w_SV param.w_SS];


% Membrane time constant 
param.tau_E = 20/1000; %ms; 20ms for E
param.tau_P = 10/1000; %ms; 10ms for all PV
param.tau_V = 10/1000; %ms; 10ms for all VIP
param.tau_S = 10/1000; %ms; 10ms for all SOM
param.tau = [param.tau_E; param.tau_P; param.tau_V; param.tau_S];


% Input
%Hh = zeros(4,1);              %input; no input = 0; somewhat larger input = 2; large input = 15
param.Hh = [0; 0; 0; 0];             %E, P, V, S

paramsave = param;

%% Functions

% ODE rate with t and u only
ode_rate2 = @(t, u, param)  (-u + param.k.*ReLU(param.W *u + param.Hh).^param.n)./param.tau;


%% Solve 4D ODE

Uu_0 = ones(4,1);           % set initial condition to 1 to calculate trajectory

dt = 0.003; %0.1; % 0.1 minutes
T0 = 0; % initial time
Tf = 5; % final time = 100 minutes
tpositions = T0:dt:Tf;

% use ode45 to numerical solve eq using Runge-Kutta 4th/5th order (variable step sizes)
[tout, x] = ode45(@(t,u) ode_rate2(t,u,param), [T0 Tf], Uu_0);

% like uniform step sizes: interpolate the result
ui = interp1(tout,x,tpositions);


figure;
subplot(2,1,1)
plot(tout, x, 'Linewidth', 2)
ylabel("rate - ode45 response")
legend("E", "P", "V", "S")
subplot(2,1,2)
plot(tpositions, ui, '-.')
xlabel("time")
ylabel("Interpolation response")
legend("E", "P", "V", "S")

%% Look at many starting positions
x0array = [0:2:80];
xarray = zeros(length(x0array),length(tpositions));

for jj = 1:length(x0array)
        [tout,x] = ode45(@(t,u) ode_rate2(t,u,param), [T0 Tf], x0array(jj));
        xarray(jj,:) = interp1(tout,x,tpositions); % store each one for plotting later
end

figure;
plot(tpositions,xarray, 'Linewidth', 1.5);
xlabel('time (minutes)');
ylabel('rate');
legend("E", "P", "V", "S")
title('rate over time; different starting points');






