% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)

%% Parameters

% Vm for neuron E and I
u_0 = [-80; 60]; %-80 for E, 60 for I
%u_0 = ones(2,1);

% Membrane time constant
tau_E = 20/1000; %ms; membrane time constant (20ms for E)
tau_I = 10/1000; %ms; membrane time constant (10ms for I)
tau = [tau_E; tau_I];

% Initial and time vector
T_init = 0;
T_final = 100;
dt = 0.001;

% Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; %ms 
% Input noise std.
sigma_0E = 0.2;     %mV; E cells
sigma_0I = 0.1;     %mV; I cells
sigma= [sigma_0E; sigma_0I]; 
sigma_scaled = sigma.*(1 + tau/tau_noise).^0.5;
eta_init = [1 -1];
seed = 1;

%% Data Structures
Tt = (T_init:dt:T_final);
Uu = zeros(length(Tt), 2);
eta = zeros(length(Tt), 2);

%% Integration
% generate a graph of fluctuations versus input
stds = [];
h_range = (0:0.2:20);

for h_factor = h_range
    
    %update input
    disp(h_factor)
    h = ones(2) * h_factor;
    
    %get time vector
    count = (1:1:length(Tt));
    t = [count; Tt]';
%     for n = t
%       X = t(n,:);
%       disp(X)
%     end

    %first generate noise vector
    eta(1,:) = eta_init;
    rng(seed);
    for iT = 1:length(Tt(:,-1))
        dt = Tt(iT + 1) - Tt(iT);
        eta(iT + 1, :) = eta(iT, :) - eta(iT,:)*dt/tau_noise + randn(1,sigma_scaled) * (2.*dt/tau_noise).^0.5;
    end

    %next, integrate neural system with noise forcint
    Uu(1,:) = u_0;
    for iT = length(Tt(:,-1))
        dt = Tt(iT + 1) - Tt(iT);
        Uu(iT +1, :) = euler(UU(iT,:), t(iT,:), dt, df) + eta(iT,:) * dt/tau;
    end
    
    append(std(Uu, 1)); %compared to Python, this std is unbiased. To change to biased, set w=1
end

plot(h_range, stds)
ylabel('std. dev. V_E/V_I')
xlabel('h')
    

%% SSN ODE (without noise term)
% 1.) open ReLU.m and ssn_ode.m function files
% 2.) use ode45 solver
[t, u] = ode45(@ssn_ode, tspan, u_0);

figure;
plot(t, u)

% 3.) now solve ODE numerical using Euler method
%clear all, close all

%% SSN with UOP process for every dt
% generate vector lenght u > ode
% add Wiener process

% Input noise std.
sigma_0E = 0.2;     %mV; E cells
sigma_0I = 0.1;     %mV; I cells
sigma_0 = [sigma_0E; sigma_0I]; 

% Initialize output d\eta (W)
W = zeros(2,length(u));


for ii = 1:length(u)-1
    W(:,ii+1) = W(:,ii) + (1./tau).*(-W(:,ii) *dt + (2 * tau .* sigma_0.^2).^0.5.*randn(2,1) * dt.^0.5);
end

plot(t, u+transpose(W))


% Difference:
% 1.) Instead of s_0, use of s_a which is the noise amplitude given for
% natural scaling
% 2.) tau_noise used as indicated in the article > far less variability of
% noise with tau_noise (ok/ not ok?)
%
% s_a = s_0.*sqrt(1 + (tau./tau_noise));
% 
% for ii = 1:length(u)-1
%     W(:,ii+1) = W(:,ii) + (1./tau_noise).*(-W(:,ii) *dt + sqrt(2 * tau_noise .* s_a.^2).*(randn(2,1) * sqrt(dt)));
% end
% 
% plot(t, u+transpose(W))