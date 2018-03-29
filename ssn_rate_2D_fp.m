%% Numerical Integration of SSN (nonlinear differential equation of one variable) 
% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)

%% Parameters
b = 2;

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

%% Plot the rates of E and I for different input h 
h = [0:1:15]; % input range
u_E = 1;%starting position  E
u_I = 1;%starting position I

% activity of E&I population
diff_E= ((-u_E + ReLU((w_EE *u_E) - (w_EI * u_I)+ h).^b) *dt) * dt./tau_E;
diff_I= ((-u_I + ReLU((w_IE *u_E) - (w_II * u_I)+ h).^b) *dt) * dt./tau_I;

figure;
hold on;
plot(h,diff_E,'g-');
plot(h,diff_I,'r-');
xlabel('input');
ylabel('rate change');
title('rate change change');
legend('E','I','Location','NorthWest');


%% Look how much E is dependent on rate of I
u_I_array = [1:1:50];

figure; hold on;
plot(h,diff_I,'r-');

for ii=1:length(u_I_array)
    diff_E= ((-u_E(ii) + ...
        ReLU((w_EE *u_E) - (w_EI * u_I)+ h).^b) *dt) * dt./tau_E;
    plot(h,diff_E,'g-');
end
xlabel('input');
ylabel('rate change');
title('rate change change');
legend('E','I','Location','NorthWest');
