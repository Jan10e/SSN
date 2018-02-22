% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


%% Parameters
k = 0.3; %scaling constant 
n = 2;
V_rest = -70; %mV; resting potential
 
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

% Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0P = 0.1;                       %mV; P cells
sigma_0S = 0.1;                       %mV; S cells
sigma_0V = 0.1;                       %mV; V cells
sigma_0 = [sigma_0E; sigma_0P; sigma_0S; sigma_0V]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(4,length(t));        % Allocate integrated eta vector

% ODE: initial 
u_0 = [-80; -60; -60; -60];                   % Vm for neuron (-80 for E, 60 for I)
u = zeros(4,length(eta));
u(:,1) = u_0;

%% Functions

% ODE
ode = @(t, u, h)  ((-u + V_rest) + W*(k.*ReLU(u - V_rest).^n) + h)./tau;


%% Generate a graph of fluctuations versus input

h_range = (0:2.5:20);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(4, length(h_range));
mean_range = zeros(4, length(h_range));
rate = zeros(4, length(h_range));
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
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
  
    
     % Get mean and std
    mean_range(:,nn) = mean(u, 2);
    stds_range(:,nn)= std(u,0,2);
    
    % Get rates
    R = k.*ReLU(u - V_rest).^n;
    rate(:,nn) = mean(R, 2);
    
end

figure;
subplot(1,3,1)
plot(h_range, rate, 'Linewidth', 2)
title("mean rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")

subplot(1,3,2)
plot(h_range, mean_range, 'Linewidth', 2)
title("mean V_E/V_I (mV)")
ylabel("mV")
xlabel("h")
legend("E", "P", "V", "S")

subplot(1,3,3)
plot(h_range, stds_range, 'Linewidth', 2)
title("std dev. V_E/V_I")
ylabel("mV")
xlabel("h")
legend("E", "P", "V", "S")


%% Voltage output for h is 0, 2, 15
figure;
h_range = [0, 2, 15];
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    subplot(1, 3, m)
    plot(t, u)
    ylabel("voltage")
    xlabel("time")
    legend("E", "P", "V", "S")

end




%% Sanity check
% look how this works for variaous tau > when tau is really small and big