% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 This script is to solve the SSN numerically without
%                           using ode45, to create a plot of time traces
%                           over a range of h


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
dt = 1e-4;
t = 0:dt:0.1;

% Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0I = 0.1;                       %mV; I cells
sigma_0 = [sigma_0E; sigma_0I]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(2,length(t));        % Allocate integrated eta vector

% ODE: initial 
u_0 = [0; 0];                   % Vm for neuron (-60 for E, -80 for I)
u = zeros(2,length(eta));
u(:,1) = u_0;

%% Functions

% ODE
ode = @(t, u, h)  ((-u + V_rest) + W*(k.*ReLU(u - V_rest).^n) + h)./tau;


%% H-range values and examine the time traces
% At what h value does the fixed point change?
h_range = (0:0.5:20);
%h_range = [0;0];  

% HOW is h-range relfected in this graph?

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
    
    
end
    
figure;
plot(t, u, 'Linewidth', 1.5)
ylabel("voltage - Forward Euler")
xlabel("time")
title("Time traces of response")
legend("E", "I")

%saveas(gcf, '2Dvolt_TimetraceHrange[-100; -80].png')




