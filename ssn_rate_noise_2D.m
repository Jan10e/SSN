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

% Connectivity Matrix W as in Hennequin
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

% Parameters - ODE: initial 
%u_0 = [-80; -60];                   % Vm for neuron (-80 for E, 60 for I)
u_0 = ones(2,1);  
u = zeros(2,length(eta));
u(:,1) = u_0;


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^n)./tau;


%% Generate a graph of fluctuations versus input (noise can be excluded; line 101)

h_range = (0:0.5:20);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(2, length(h_range));
mean_range = zeros(2, length(h_range));
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
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; %set eta * 0 to remove noise
    end
  
    
     % Get mean and std
    mean_range(:,nn) = mean(u, 2);
    stds_range(:,nn)= std(u,0,2);
    
end

figure;
subplot(1,2,1)
plot(h_range, mean_range, 'Linewidth', 1.5)
title("mean rate")
ylabel("rate")
xlabel("h")
legend("E", "I")


subplot(1,2,2)
plot(h_range, stds_range, 'Linewidth', 1.5)
title("std dev. rate")
ylabel("rate")
xlabel("h")
legend("E", "I")


%% Get Noise Correlations (NC)
%High NC at what rate and h
[ncHigh_E, ncHigh_idxE]= max(stds_range(1,:));
[ncHigh_I, ncHigh_idxI] = max(stds_range(2,:));

ncHigh_hE = h_range(ncHigh_idxE);
ncHigh_hI = h_range(ncHigh_idxI);


% Smallest NC, starting at the high NC of matrix to avoid low NC between h=0-2
[ncLow_E, ncLow_idxE]= min(stds_range(1,ncHigh_idxE:end));
[ncLow_I, ncLow_idxI]= min(stds_range(2,ncHigh_idxI:end));

ncLow_hE = h_range(ncLow_idxE);
ncLow_hI = h_range(ncLow_idxI);


%% Get states

% h_spont 
%is where the nc is highest, ncHigh, which is at h = 2
%constant is ncHigh_hE as in the vector the value for E is 1
ncHigh = ncHigh_hE;

% get value for I in vector
I_spont = ncHigh_hI/ncHigh_hE;
h_spont = ncHigh*[1;I_spont];


%h_stim 
%is where nc is lowest, ncLow, which is at h = 15
%constant is ncLow_hE as in the vector the value for E is 1
ncLow = ncLow_hE;

%get value for I in vector
I_stim = ncLow_hI/ncLow_hE;
h_stim = ncLow * [1;I_stim];


%h_att
%NC decreases more for h_stim, so less fluctuation in rates and evoked rates
%increase (h > 15)
alpha = 20;
h_att = alpha * [1;1]; %play around with I value


%h_loc
% NC decreases more for h_stim, so less fluctuation in rates and evoked rates
%increase (h > 15). This is a large effect compared to h_att
lambda = 30;
h_loc = lambda * [1;0.5]; %play around with I value


%% Rate output for h is h_spont, h_stim, h_att, h_loc
figure;
h_range = [h_spont, h_stim, h_att, h_loc];
for m = 1:length(h_range)
    
    % update input
    h = h_range(:,m);
    disp(h)
    
    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    subplot(1, 4, m)
    plot(t, u, 'Linewidth',1) %order plots: h_spont, h_stim, h_att, h_loc
    ylim([0 250])
    ylabel("rate")
    xlabel("time")
    legend("E", "I")

end


%% Get mean and std dev of different h's

