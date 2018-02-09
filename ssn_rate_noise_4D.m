% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


%% Parameters
k = 0.01; %scaling constant 
n = 2.2;

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
tau_S = 10/1000; %ms; 10ms for all SOM
tau_V = 10/1000; %ms; 10ms for all VIP
tau = [tau_E; tau_P; tau_S; tau_V];

% Time vector
dt = 1e-3;
t = 0:dt:100;

% Parameters - Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0P = 0.1;                       %mV; P cells
sigma_0S = 0.1;                       %mV; S cells
sigma_0V = 0.1;                       %mV; V cells
sigma_0 = [sigma_0E; sigma_0P; sigma_0S; sigma_0V]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(4,length(t));        % Allocate integrated eta vector

% Parameters - ODE: initial 
%u_0 = [-80; -60; -60; -60];                   % Vm for neuron (-80 for E, 60 for I)
u_0 = ones(4,1);  
u = zeros(4,length(eta));
u(:,1) = u_0;


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^n)./tau;


%% Generate a graph of fluctuations versus input (noise can be excluded; line 101)

h_range = (0:2.5:20);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(4, length(h_range));
mean_range = zeros(4, length(h_range));
for nn = 1:length(h_range)
    
    % update input
    h_factor = h_range(nn);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(4,1))) *(1./tau_noise);
    end
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + (eta(:,ii)*0) * dt./tau; %set eta * 0 to remove noise
    end
  
    
     % Get mean and std
    mean_range(:,nn) = mean(u, 2);
    stds_range(:,nn)= std(u,0,2);
    
end

figure;
subplot(1,2,1)
plot(h_range, mean_range)
title("mean rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")


subplot(1,2,2)
plot(h_range, stds_range)
title("std dev. rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")


%% Sanity check
% look how this works for variaous tau > when tau is really small and big

% small tau (1/1000) doesn't affect the ode too much, however changing it
% to a really big tau (100000/1000) affects all cells, with S least
% affected

%% Recreate fig 7c of Kuchibotla
% create separate input to cells and look at change from baseline (h = 0)

figure;
%h_range = [0, 2, 15];
h_range = [0; 15; 0; 0]; %E, P, V, S
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + (eta(:,ii)*0) * dt./tau;  %first try without noise (eta * 0)
    end
    
    subplot(1, 4, m)
    plot(t, u, 'LineWidth',2)
    ylabel("rate")
    xlabel("time")
    legend("E", "P", "V", "S")

end

% h = 0 for all: shows that all rates go from 1 to 0 within 0.1ms (E a bit later)

% h = 1 for all: rates of I's go from 1 to 0.01 in 0.05ms; for E go from 1 to
% 0.01 in 0.1ms

% h_E = 15 (all other h = 0): E (rate =~1.6);  P (rate = ~ 2.5); V (rate = ~ 3.5); S (rate = 4.8)// (different compared to ssn_noise_4D.m)

% h_V = 15 (all other h = 0): E (rate = 1.6);  P (rate = ~ 2.5); V (rate = ~ 3.5); S (rate = 4.8)// (different compared to ssn_noise_4D.m; also similar to h_E only (which is postulated by Kuchibotla that h_E only and h_V is the same))

% h_P = 15: same as h_E and h_V
