% author: Jantine Broek
% collaborator: Yashar Ahmadian
% goal: recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. This is the circuit dynamics part 
% model: Reduced rate model (See experimental procedures)

%% Mean potential Vm
% The more depolarized the cell, the higher the spike rate

% Generate random Vm for monotary population averaging of all E or I cells
% Excitory cell Vm
% range --40 and -30
E_min = -40;
E_max = -30;
V_E = (E_max - E_min).*rand(100,1) + E_min; %V_m vector for neuron population

% Inhibitory cell Vm
% make sure the firing rate of I is not too high (so more depolarized),
% because otherwhise it would shut E down
I_min = -45;
I_max = -40;
V_I = (I_max - I_min).*rand(100,1) + I_min; %V_m vector for neuron population


%% Momentarily firing rate r_j(t)
% r_i(t) = k[V_i(t) - V_{rest}]^n_+

k = 0.3; %scaling constant 
n = 2; 

% create constraint data where rate is [V_i(t) - V_{rest}] >= 0
% Excitatory rate: r_E
r_Etemp =[];
r_E = zeros(size(V_E));

for ii = 1:length(r_E)
    r_Etemp(ii) = V_E(ii) - V_rest;
    
    if r_Etemp(ii) <= 0
        r_E(ii) = 0;
    else
        r_E(ii) = (k*r_Etemp(ii))^n; %multiply by k for existing values      
    end
end


% Inhibitory rate: r_I
r_Itemp = [];
r_I = zeros(size(V_I));

for ii = 1:length(r_I)
    r_Itemp(ii) = V_I(ii) - V_rest;
    
    if r_Itemp(ii) <= 0
        r_I(ii) = 0;
    else
        r_I(ii) = (k*r_Itemp(ii))^n; %multiply by k for existing values
    end
end

%% Multiply W with rate and sum over all cells
% Connectivity Matrix W
W_EE = 1.25;
W_EI = 0.65;
W_IE = 1.2;
W_II = 0.5;
W = [W_EE -W_EI; W_IE -W_II];

% Sum over all Excitatory cells
Vdyn_E = zeros(size(W));
for jj = 1:size(r_E,1)
       Vdyns_E = W.*r_E(jj);
       Vdyn_E = Vdyn_E + Vdyns_E;
end
clear Vdyns_E

% Sum over all Inhibitory cells
Vdyn_I = zeros(size(W));
for jj = 1:size(r_I,1)
       Vdyns_I = W.*r_I(jj);
       Vdyn_I = Vdyn_I + Vdyns_I;
end
clear Vdyns_I


%% Input drive h
% h_i(t) = potentially time-varying but deterministic component of external
% input. Set this?


%% Input noise eta - Ornstein-Uhlenbeck process
% eta_i(t) = input noise modelled as a multivariate Ornstein-Uhlenbeck process

tau_noise = 50; %ms, external noise time constant


%% Circuit dynamics
% 	\frac{dV_i}{dt} = \frac{1}{\tau_i} (-V_i + V_{rest} + \sum_{j \in E cells} W_{ij} k[V_j - V_0]^n_+ 
%  - \sum_{j \in I cells} W_ij k[v_j - v_0]^n_+ + h_i(t) ) + \eta_i(t)

tau_i = 20; %ms; membrane time constant (20ms for E and 10ms for I)
V_rest = -70; %mV; resting potential
t_bin = 30; %ms; time bins

dVi = (1/tau_i) * (-Vi + V_rest + Vdyn_E - Vdyn_I + ht) + eta_t;  

%dVi = (1/tau_i) * (-V_rest + Vdyn_E - Vdyn_I);  

