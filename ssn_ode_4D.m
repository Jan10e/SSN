function [ du ] = ssn_ode_4D(t, u)
%SSN ODE for four cell populations 
%   4 cell populations: E, I_1, I_2, and I_3, evaluated with a ReLU function; t is time bins and u is
%   Vm of cell i

%% Parameters
k = 0.3; %scaling constant 
n = 2;
V_rest = -70; %mV; resting potential
 
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
tau_S = 10/1000; %ms; 10ms for all SOM
tau_V = 10/1000; %ms; 10ms for all VIP
tau = [tau_E; tau_P; tau_S; tau_V];

%input
h = [0; 0; 0; 0]; %mV; no input = 0; somewhat larger input = 2; large input = 15

%% ODE
du = ((-u + V_rest) + W*(k * ReLU(u - V_rest).^n) + h)./tau;

end

