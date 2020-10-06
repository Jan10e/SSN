% author: Jantine Broek
% date: Dec 2017
% Goal: Simulating the Ornstein-Uhlenbeck process
% Reference: https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
%% 
clear
clc

%% Paths
dir_base = '/Users/jantinebroek/Documents/03_projects/02_SSN/ssn_nc_attention';

dir_work = '/matlab';
dir_data = '/data';
dir_fig = '/figures';

cd(fullfile(dir_base, dir_work));

%% Simulation
th = 1;
mu = 1.2;
sig = 0.3;
dt = 1e-2;
t = 0:dt:2;                         % Time vector
x = zeros(1,length(t));             % Allocate output vector, set initial condition
rng(1);                             % Set random seed

for i = 1:length(t)-1
    x(i+1) = x(i)+th*(mu-x(i))*dt+sig*sqrt(dt)*randn;
end

f1 = figure;
plot(t,x);
hold on 

%% Solution in terms of integral
th = 1;
mu = 1.2;
sig = 0.3;
dt = 1e-2;
t = 0:dt:2;                      % Time vector
x0 = 0;                          % Set initial condition
rng(1);                          % Set random seed
W = zeros(1,length(t));          % Allocate integrated W vector

for i = 1:length(t)-1
    W(i+1) = W(i)+sqrt(dt)*exp(th*t(i))*randn; 
end

ex = exp(-th*t);
x = x0*ex+mu*(1-ex)+sig*ex.*W;


f2 = figure;
subplot(2,1,1)
plot(t,ex);
title('deterministic signal')
subplot(2,1,2)
plot(t,x);
title('signal with noise')

%% Analytical solution
% with Wiener increments and ``scaled-time transformed'' Wiener process
th = 1;
mu = 0;
sig = 0.3;
dt = 1e-2;
t = 0:dt:5;                             % Time vector
x0 = 1;                                 % Set initial condition
rng(1);                                  % Set random seed
W = zeros(1,length(t));        % Allocate integrated W vector

for i = 1:length(t)-1
    W(i+1) = W(i)+sqrt(exp(2*th*t(i+1))-exp(2*th*t(i)))*randn;
end

ex = exp(-th*t);
x = x0*ex+mu*(1-ex)+sig*ex.*W/sqrt(2*th);

f3 = figure;
subplot(2,1,1)
plot(t,ex);
title('deterministic signal')
subplot(2,1,2)
plot(t,x);
title('signal with noise')

%% Export/Save
outfile = 'OrnsteinUhlenbeck_';
       
suffix_fig_f1 = 'simulation';
suffix_fig_f2 = 'integral_soln';
suffix_fig_f3 = 'Wiener_increments';
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


