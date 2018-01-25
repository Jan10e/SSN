% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)

%% Parameters
u_0 = [-80; 60]; %-80 for E, 60 for I
%u_0 = ones(2,1);

% tspan = [0 5];
dt = 0.0003;
tspan =(0:dt:5); % 30ms time bin

%% SSN ODE
[t, u] = ode45(@ssn_ode, tspan, u_0);

figure;
plot(t, u)

%% Generate Gaussian Stochastic process
% generate vector lenght u > ode
% add Wiener process

tau_E = 20; %ms; membrane time constant (20ms for E)
tau_I = 10; %ms; membrane time constant (10ms for I)
tau = [(tau_E/1000); (tau_I/1000)];

s_0E = 0.2;     % input noise std. for E cells
s_0I = 0.1;     % input noise std. for I cells
s_0 = [s_0E; s_0I]; 

W = zeros(2,length(u));
%dE = randn(length(u) * dt);


for ii = 1:length(u)-1
    W(:,ii+1) = W(:,ii) + (1./tau).*(-W(:,ii) *dt + (2 * tau .* s_0.^2).^0.5.*randn(2,1) * dt.^0.5);
end

plot(t, u+transpose(W))


