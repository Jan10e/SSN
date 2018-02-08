% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)    

%% Solve SSN ODE (without noise term) 
% 1.) open ReLU.m and ssn_ode.m function files
% 2.) define parameters

% Vm for neuron E and I
%u_0 = [-80; 60]; %-80 for E, 60 for I
u_0 = ones(2,1);           % set initial condition to 1 to calculate trajectory

% Time vector
tspan = (0:0.003:5);


% 3.) use ode45 to numerical solve eq using Runge-Kutta 4th/5th order
[t, u] = ode45(@ssn_ode, tspan, u_0);

figure;
plot(t, u)

x = u(1,:);
y = u(2,:);


%% Find Fixed Points



%% SSN with UOP process for every dt
% generate vector lenght u > ode
% add Wiener process

% Membrane time constant 
tau_E = 20/1000; %ms; 20ms for E
tau_I = 10/1000; %ms; 10ms for I
tau = [tau_E; tau_I];

% Input noise std.
sigma_0E = 0.2;     %mV; E cells
sigma_0I = 0.1;     %mV; I cells
sigma_0 = [sigma_0E; sigma_0I]; 

% Initialize output d\eta (W)
W = zeros(2,length(u));

for ii = 1:length(u)-1
    W(:,ii+1) = W(:,ii) + (1./tau).*(-W(:,ii).*dt + (2 * tau .* sigma_0.^2).^0.5.*randn(2,1) * dt.^0.5);
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