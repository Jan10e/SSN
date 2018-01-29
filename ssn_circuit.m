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

% Time span
%tspan = [0 5];
dt = 0.0003;
tspan =(0:dt:3); % 30ms time bin

% Noise correlation time constant
tau_noise = 50/1000; %ms 

% Membrane time constant
tau_E = 20; %ms; membrane time constant (20ms for E)
tau_I = 10; %ms; membrane time constant (10ms for I)
tau = [(tau_E/1000); (tau_I/1000)];

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
s_0E = 0.2;     %mV; E cells
s_0I = 0.1;     %mV; I cells
s_0 = [s_0E; s_0I]; 

% Initialize output d\eta (W)
W = zeros(2,length(u));


for ii = 1:length(u)-1
    W(:,ii+1) = W(:,ii) + (1./tau).*(-W(:,ii) *dt + (2 * tau .* s_0.^2).^0.5.*randn(2,1) * dt.^0.5);
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