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
tau_V = 10/1000; %ms; 10ms for all VIP
tau_S = 10/1000; %ms; 10ms for all SOM
tau = [tau_E; tau_P; tau_V; tau_S];

% Time vector
%dt = 1e-4;
%t = 0:dt:5;

dt = 0.003; %0.1; % 0.1 minutes
T0 = 0; % initial time
Tf = 5; % final time = 100 minutes
tpositions = T0:dt:Tf;

% Parameters - Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0P = 0.1;                       %mV; P cells
sigma_0V = 0.1;                       %mV; V cells
sigma_0S = 0.1;                       %mV; S cells
sigma_0 = [sigma_0E; sigma_0P; sigma_0V; sigma_0S]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(4,length(tpositions));        % Allocate integrated eta vector

% Parameters - ODE: initial 
%u_0 = [-80; -60; -60; -60];                   % Vm for neuron (-80 for E, 60 for I)
u_0 = ones(4,1);  
u = zeros(4,length(eta));
u(:,1) = u_0;


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^n)./tau;


%% Generate a graph of fluctuations versus input (noise can be excluded)
h_range = (0:2.5:50);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(4, length(h_range));
mean_range = zeros(4, length(h_range));
for nn = 1:length(h_range)
    
    % update input
    h_factor = h_range(nn);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Generate noise vector
    for ii = 1:length(tpositions)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(4,1))) *(1./tau_noise);
    end
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(tpositions, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; %set eta * 0 to remove noise
    end
  
    
     % Get mean and std
    mean_range(:,nn) = mean(u, 2);
    stds_range(:,nn)= std(u,0,2);
    
end

figure;
subplot(1,2,1)
plot(h_range, mean_range, 'Linewidth', 2)
title("mean rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")


subplot(1,2,2)
plot(h_range, stds_range, 'Linewidth', 2)
title("std dev. rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")

saveas(gcf,'4D_meanstd_noise.png')
%saveas(gcf, '4D_bigTau.png')


%% Sanity check
% look how this works for variaous tau > when tau is really small and big

% small tau (4/1000 for E; 2/1000 for I) doesn't affect the ode too much, however changing it
% to a really big tau (10000/1000) affects all cells, with S least
% affected

% Didn't change tau noise (50/1000)


%% Different starting positions
% already done in ssn_rate_4D_solve.m This is just to check the outcomes
Tf = 0.5; % final time = 100 minutes
tpositions = T0:dt:Tf;

h = [0; 0; 0; 0];

u0array = repmat([-150:20:100],4,1);
uarray = zeros(length(u0array),length(tpositions),4);

%u = zeros(4, length(tpositions));

for jj=1:length(u0array)
    u(1) = u0array(:,jj); % starting position for u; updates each time

    for ii=2:length(tpositions)
        u(:,ii) = u(:,ii-1) + ode_rate(tpositions(ii), (u(:,ii-1)), h) *dt + (eta(:,ii-1)*0) * dt./tau;
        %urow(:,ii) = u(ii);
    end
    uarray(jj,:) = u; % store each one for plotting later
    
end

% figure;
% plot(tpositions,xarray,'r-');
% xlabel('Time (minutes)');
% ylabel('Concentration (uM)');
% title('Concentration over time; different starting points');

%% Rate output for h is 0, 2, 15
figure;
h_range = [0, 2, 15];
u = zeros(4, length(tpositions));
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(tpositions, (u(:,ii)), h)*dt + eta(:,ii)* dt./tau; 
    end    
    
    %plot for every h the rates in one plot
    subplot(1, 3, m)
    plot(tpositions, u) 
    ylabel("rate")
    xlabel("time")
    legend("E", "P", "V", "S")

end

%% Recreate fig 7c of Kuchibotla
% create separate input to cells and look at change from baseline (h = 0)

h_range = [0; 15; 0; 15]; %E, P, V, S (passive state: E and V; active state: P and S)
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = h_range;

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(tpositions, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau;  %first try without noise (eta * 0)
    end


end


figure;
plot(tpositions, u, 'LineWidth',2)
ylabel("rate")
xlabel("time")
legend("E", "P", "V", "S")

% h = 0 for all: shows that all rates go from 1 to 0 within 0.1ms (E is a bit slower)

%saveas(gcf, 'passive_context.png')
%saveas(gcf, 'active_context.png')

%% Passive vs Active

% passive state is stimulus only. As discussed with Yashar, stimulus only
% is h input to E and PV cells, although h does not have to be similar.

% In Kuchibhotla, passive state is increase of activity by Exc and VIP cells 
% i.e. h = 15 for E and h = 15 for VIP cells

% % run script with h = 15 for E cells and h = 15 for VIP
u_passive = u;

%Active state in Kuchibhotla is described as increased activity of PV and SOM. Set
% h=15 for PV and SOM

% run script with h = 15 only for E cells
u_active = u;



%% Modulation index 

% values from Kuchibhotla fig 3.
E_pass = 67;
E_act = 33;
P_pass = 39;
P_act = 61;
V_pass = 78;
V_act = 22;
S_pass = 40;
S_act = 60;

val = [E_pass E_act;
P_pass P_act;
V_pass V_act;
S_pass S_act;];

mod = zeros(length(val),1);
for n = 1:length(val)
    mod(n) = (val(n,2) - val(n,1))/val(n,2);
end


%values obtained with simulation 
act_mean = mean(u_active, 2);
pas_mean = mean(u_passive, 2);

val_sim = [pas_mean act_mean];

mod_sim = zeros(length(val_sim),1);
for m = 1:length(val)
    mod_sim(m) = (val_sim(m,2) - val_sim(m,1))/val_sim(m,2);
end


%% Give h-range input to I's only
% create separate input to cells and look at change from baseline (h = 0)

h_range = [15; 15; 15; 15]; %E, P, V, S (passive state: E and V; active state: P and S)
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = h_range;

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(tpositions, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau;  %first try without noise (eta * 0)
    end

end


figure;
plot(tpositions, u, 'LineWidth',2)
ylabel("rate")
xlabel("time")
legend("E", "P", "V", "S")

