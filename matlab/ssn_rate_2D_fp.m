%% Numerical Integration of SSN (nonlinear differential equation of one variable) 
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
diff_E= ((-u_E + functions.ReLU((w_EE *u_E) - (w_EI * u_I)+ h).^b) *dt) * dt./tau_E;
diff_I= ((-u_I + functions.ReLU((w_IE *u_E) - (w_II * u_I)+ h).^b) *dt) * dt./tau_I;

f1 = figure;
hold on;
plot(h,diff_E,'g-');
plot(h,diff_I,'r-');
xlabel('input');
ylabel('rate change');
title('rate change change');
legend('E','I','Location','NorthWest');


%% Look how much E is dependent on rate of I
u_I_array = [1:1:50];

f2 = figure; hold on;
plot(h,diff_I,'r-');

for ii=1:length(u_I_array)
    diff_E= ((-u_E(ii) + ...
        functions.ReLU((w_EE *u_E) - (w_EI * u_I)+ h).^b) *dt) * dt./tau_E;
    plot(h,diff_E,'g-');
end
xlabel('input');
ylabel('rate change');
title('rate change change');
legend('E','I','Location','NorthWest');

%% Export/Save
outfile = 'SSN_rate_2D_fp_';
       
suffix_fig_f1 = 'rate_change_h';
suffix_fig_f2 = 'rate_change_E_dependent_I';
suffix_data = '';       

out_mat = [outfile, suffix_data, '.mat'];
out_fig_f1_png = [outfile, suffix_fig_f1, '.png'];
out_fig_f2_png = [outfile, suffix_fig_f2, '.png'];

outpath_data = fullfile(dir_base, dir_data, out_mat);
outpath_fig_f1_png = fullfile(dir_base, dir_fig, out_fig_f1_png);
outpath_fig_f2_png = fullfile(dir_base, dir_fig, out_fig_f2_png);

% figures
saveas(f1, outpath_fig_f1_png,'png')
saveas(f2, outpath_fig_f2_png,'png')

