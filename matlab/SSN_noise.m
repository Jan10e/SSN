% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)
%% 
clear
clc

%% Paths
dir_base = '/Users/jantinebroek/Documents/03_projects/02_SSN/ssn_nc_attention';

dir_work = '/matlab';
dir_data = '/data';
dir_fig = '/figures';

cd(fullfile(dir_base, dir_work));

%% OUP for noise
% with Wiener increments and ``scaled-time transformed'' Wiener process
% This is a Forcing term for the dynamical model

th = 1;
mu = 0;                                 % mu to 0 as it needs to decay to 0
sig = 0.3;
dt = 1e-3;
t = 0:dt:100;                             % Time vector
x0 = 1;                                 % Set initial condition (not 0, otherwise you don't see it decaying to 0)
rng(1);                                  % Set random seed

% create two processes: one for E population and one for I population
W = zeros(2,length(t));        % Allocate integrated W vector

for ii = 1:length(t)-1
    W(:,ii+1) = W(:,ii)+sqrt(exp(2*th*t(:,ii+1))-exp(2*th*t(:,ii)))*randn(2,1);
end

ex = exp(-th*t);
x = x0*ex+mu*(1-ex)+sig*ex.*W/sqrt(2*th);

f1 = figure;
plot(t,x);

%% Euler method forward method + noise W

% Vm for neuron E and I
u_0 = [-60; -80]; %-80 for E, 60 for I
%tend = 1;

h = [0; 0];

% Euler loop
u = zeros(2,length(W));
u(:,1) = u_0;
for ii = 1: length(W)-1
    
      % Take the Euler step + x(i) which is the noise
%       u(:,ii+1) = u(:,ii) + functions.ssn_rate_ode(t, (u(:,ii) + x(:,ii)), h)*dt;
        u(:,ii+1) = u(:,ii) + functions.ssn_rate_ode(t, (u(:,ii) + x(:,ii)))*dt;
      
end

f2 = figure;
plot(t, u)


%% Getting mean and stds
h_range = (0:2.5:20);
%h_range = 1;
stds_range = zeros(2, length(h_range));
mean_range = zeros(2, length(h_range));
for n = 1:length(h_range)
    
    % update input
    h_factor = h_range(n);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    
    %get time vector
%     count = (1:1:length(u));
%     t = [count; u]';
 
    %Generate noise vector
    for ii = 1:length(t)-1
        W(:,ii+1) = W(:,ii)+sqrt(exp(2*th*t(:,ii+1))-exp(2*th*t(:,ii)))*randn(2,1);
    end

    %Integrate neural system with noise forcing
    for ii = 1: length(W)-1  
      % Take the Euler step + x(i) which is the noise
%       u(:,ii+1) = u(:,ii) + functions.ssn_rate_ode(t, (u(:,ii) + x(:,ii)), h)*dt;
      u(:,ii+1) = u(:,ii) + functions.ssn_rate_ode(t, (u(:,ii) + x(:,ii)))*dt;
    end
    
     % Get mean and std
    mean_range(:,n) = mean(u, 2);
    stds_range(:,n)= std(u,0,2);
    
      
end

f3 = figure;
plot(t, u)

plot(h_range, mean_range)
plot(h_range, stds_range)


%% Sanity check
% look how this works for variaous tau > when tau is really small and big

%% Export/Save
outfile = 'SSN_noise_';
       
suffix_fig_f1 = 'OUP_noise';
suffix_fig_f2 = 'euler_with_OUnoise';
suffix_fig_f3 = 'mean_std';
suffix_data = '';       

out_mat = [outfile, suffix_data, '.mat'];
out_fig_f1_png = [outfile, suffix_fig_f1, '.png'];
out_fig_f2_png = [outfile, suffix_fig_f2, '.png'];
out_fig_f3_png = [outfile, suffix_fig_f3, '.png'];

outpath_data = fullfile(dir_base, dir_data, out_mat);
outpath_fig_f1_png = fullfile(dir_base, dir_fig, out_fig_f1_png);
outpath_fig_f2_png = fullfile(dir_base, dir_fig, out_fig_f2_png);
outpath_fig_f3_png = fullfile(dir_base, dir_fig, out_fig_f3_png);

% figures
saveas(f1, outpath_fig_f1_png,'png')
saveas(f2, outpath_fig_f2_png,'png')
saveas(f3, outpath_fig_f3_png,'png')
