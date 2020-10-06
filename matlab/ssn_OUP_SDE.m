% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 Model an Ornstein-Uhlenbeck Stochastic Diff Eq
%                           (SDE) to model the noise term, eta, in
%                           Hennequin's model
% model:              Ornstein-Uhlenbeck Stochastic Diff Eq (OU-SDE)
%% 
clear
clc

%% Paths
dir_base = '/Users/jantinebroek/Documents/03_projects/02_SSN/ssn_nc_attention';

dir_work = '/matlab';
dir_data = '/data';
dir_fig = '/figures';

cd(fullfile(dir_base, dir_work));

%% Parameters for OUP
% Time constant
tau_noise = 50;     % noise correlation time constant

% Input noise Std
% s_0E = 0.2;     % input noise std. for E cells
% s_0I = 0.1;     % input noise std. for I cells
s_0E = 1.;     % input noise std. for E cells
s_0I = 0.5;     % input noise std. for I cells
s_0 = [s_0E; s_0I]; 

% Membrane time constant
tau_E = 20; %ms; membrane time constant (20ms for E)
tau_I = 10; %ms; membrane time constant (10ms for I)
tau = [(tau_E/1000); (tau_I/1000)];

%% Covariance matrix
% step 1: create noise amplitude for E and I to model the variance of the
% noise
s_a = s_0.*sqrt(1+(tau/tau_noise));

% step 2: add the noise amplitude to the noise term Sigma^(noise)
%d_ij =1 if i = j (feedforward?) and 0 otherwise (recurrent?)
% as W = [w_EE w_EI; w_IE w_II]; , set EE and II to 0 and EI and IE to 1
delta = [0 1; 1 0];

sum_noise = s_a.^2.*delta;

%% Create drift and diffusion function
F = @(t, eta) -eta * (1/tau_noise); %drift
G = @(t, eta) sqrt(2 * tau_noise * sum_noise) * (1/tau_noise); %diffusion

%% SDE
eta = sde(F, G, 'StartState', [0; 1]); 
