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
 
% Connectivity Matrix W
w_EE = 1.25;
w_EI = -0.65;
w_IE = 1.2;
w_II = -0.5;
W = [w_EE w_EI; w_IE w_II];

% Membrane time constant 
tau_E = 20/1000;                   %ms; 20ms for E
tau_I = 10/1000;                    %ms; 10ms for I
tau = [tau_E; tau_I];

% Time vector
dt = 1e-3;
t = 0:dt:100;

% Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0I = 0.1;                       %mV; I cells
sigma_0 = [sigma_0E; sigma_0I]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(2,length(t));        % Allocate integrated eta vector

% ODE: initial 
u_0 = [-60; -80];                   % Vm for neuron (-80 for E, 60 for I)
u = zeros(2,length(eta));
u(:,1) = u_0;

%% Functions

% ODE
ode = @(t, u, h)  ((-u + V_rest) + W*(k.*ReLU(u - V_rest).^n) + h)./tau;


%% Generate a graph of fluctuations versus input

h_range = (0:0.5:20);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(2, length(h_range));
mean_range = zeros(2, length(h_range));
rate = zeros(2, length(h_range));
for nn = 1:length(h_range)
    
    % update input
    h_factor = h_range(nn);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    

    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
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
legend("E", "I")

subplot(1,3,2)
plot(h_range, mean_range, 'Linewidth', 2)
title("mean V_E/V_I (mV)")
ylabel("mV")
xlabel("h")
legend("E", "I")

subplot(1,3,3)
plot(h_range, stds_range, 'Linewidth', 2)
title("std dev. V_E/V_I")
ylabel("mV")
xlabel("h")
legend("E", "I")

%% Voltage output for h is 0, 2, 15
figure;
h_range = [0, 2, 15];
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    subplot(1, 3, m)
    plot(t, u, 'Linewidth',1)
    ylabel("voltage")
    xlabel("time")
    legend("E", "I")

end

%saveas(gcf, '2Dvolt_h0215.png')


%% Sanity check
% look how this works for variaous tau > when tau is really small and big