% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 Get the spikes from the mean-field model and
%                          calculate the variance to look for changes in noise correlation when
%                          brain state changes
% model:               SSN (mean-field) rate model

% step 1:   get spikes from rate model
% step 2:   calculate variance with obtained spikes
% step 3:   calculate Fano factors
% step 4:   run simulations for different brain states



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
tau_noise = 10/1000; 
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

% Transient location
trans = find(t == 0.3);

% Parameters spikes
s_0 = ones(2,1);
s = zeros(2, length(u(trans:end)));
s(:,1) = s_0;


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^n)./tau;


%% Get rates
h_range = (0:0.5:15);
rates = zeros(length(h_range), 2, length(t));
for n = 1:length(h_range)
    
    % update input
    h_factor = h_range(n);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    
    %Generate noise vector
    for i = 1:length(t)-1
        eta(:,i+1) = eta(:,i) + (-eta(:,i) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
    end
    
    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    %output
    rates(n,:,:) = u; 
        
end


%% Remove negative rates and transient

%negative values to 0 (avoid negative spikes)
rates(rates<0) = 0; 

%identify transient (~300ms)
trans = find(t == 0.3);
ratesT = rates(:,:,trans:end);
tt = t(trans:end);

%ratesE = squeeze(ratesT(:,1,:));
%ratesI = squeeze(ratesT(:,2,:));


%% Spikes: Integrate with stretching window (running area)
% \delta n_i = \int_0^T \delta r_i(t) dt, with T = time constant

% window/step size: 200ms
step = 0.2/dt;

clear x y s 

% Integrate rates to get spikes
for i = 1:size(ratesT,1)
    
    % update input
    disp(i)
    
    % numerical integrate with stretching window
    x = tt(1:step:end);
    y = ratesT(i,:,1:step:end);
    for j = 1:size(ratesT,2)
         for k = 2:length(x)
             s(:,j,k-1) = trapz(x(1:k),y(:,j,1:k));
         end 
    end
     spike_strch(i,:,:) = s;
end

spikeE = spike_strch(:,1,:);


subplot(221),plot(x,y)
subplot(222),plot(x(2:end),s(1,1,:))

subplot(223),plot(ratesT(1,1,:))
subplot(224),plot(spike_strch(1,1,:))






%% Spikes: Integrate with sliding window

% window/step size: 200ms
step = 0.2/dt;

clear x y s

% number of steps
num_stps = round(length(ratesT)/step)-1;

spike_stps =zeros(1,num_stps+1);
spike_h = zeros(size(ratesT,1), num_stps+1);
for i = 1:size(ratesT,1)
    
    disp(i)
    
    for j = 1:num_stps+1
    
        disp(j)
    
       % numerical integrate with sliding window
        x = tt(j:j+step-1);
        y = ratesT(i,:,j:j+step-1);
        for k =1:size(ratesT,2)
            s(:,k,:) = trapz(x,y(:,k,:));
        end
        
        % collect every step integration
        spike_stps(:,:,j) = s;
    end
    
    % collect every h-input
    spike_h(i,:,:) = spike_stps;
    
end

subplot(211),plot(ratesT(:,1,43))
subplot(212),plot(spike_h(:,1,43))



%% Variance
% double integral for two time point over time constant
