function [ du ] = ssn_ode(u, t)
%SSN ODE is a generic function 
%   Is working with ReLU function and parameters; t is time bins and u is
%   Vm of cell i

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
tau_E = 20/1000; %ms; 20ms for E
tau_I = 10/1000; %ms; 10ms for I
tau = [tau_E; tau_I];

%input
h = [0; 0]; %mV; no input = 0; somewhat larger input = 2; large input = 15

%% ODE
du = ((-u + V_rest) + W*(k * ReLU(u - V_rest).^n) + h)./tau;

end

