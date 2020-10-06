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
k = 0.01; %scaling constant 
% n = 2.2;
%k = 0.3; %scaling constant as in Hennequin
n = 2;

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
tau_V = 10/1000; %ms; 10ms for all VIP
tau_S = 10/1000; %ms; 10ms for all SOM
tau = [tau_E; tau_P; tau_V; tau_S];

% Time vector
dt = 1e-3;
t = 0:dt:100;

% dt = 0.003; %0.1; % 0.1 minutes
% T0 = 0; % initial time
% Tf = 5; % final time = 100 minutes
% t = T0:dt:Tf;

% Parameters - Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
sigma_0E = 0.2;                      %mV; E cells
sigma_0P = 0.1;                       %mV; P cells
sigma_0V = 0.1;                       %mV; V cells
sigma_0S = 0.1;                       %mV; S cells
sigma_0 = [sigma_0E; sigma_0P; sigma_0V; sigma_0S]; %input noise std
sigma_a = sigma_0.*sqrt(1 + (tau./tau_noise));
eta = zeros(4,length(t));        % Allocate integrated eta vector

% Parameters - ODE: initial 
%u_0 = [-80; -60; -60; -60];                   % Vm for neuron (-80 for E, 60 for I)
u_0 = ones(4,1);  
u = zeros(4,length(eta));
u(:,1) = u_0;


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*functions.ReLU(W *u + h).^n)./tau;


%% Generate a graph of fluctuations versus input (noise can be excluded)
h_range = (0:0.5:15);
stds_range = zeros(4, length(h_range));
mean_range = zeros(4, length(h_range));
mx = zeros(length(h_range),4, length(t));
for i = 1:length(h_range)
    
    % update input  
    h_factor = h_range(i);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(4,1))) *(1./tau_noise);
    end
    
    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau;
    end
    
    
    % add u to matrix 
     par_change(i,:,:) = u;
    
     % Get mean and std
    mean_par(:,i) = mean(u, 2);
    stds_par(:,i)= std(u,0,2);
    
end

% save('data/4Drate_par_change_eq.mat', 'par_change')
% save('data/4Drate_mean_eq.mat', 'mean_par')
% save('data/4Drate_stds_eq.mat', 'stds_par')


%% Plots
figure;
subplot(1,2,1)
plot(h_range, mean_par, 'Linewidth', 2)
title("mean rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")


subplot(1,2,2)
plot(h_range, stds_par, 'Linewidth', 2)
title("std dev. rate")
ylabel("rate")
xlabel("h")
legend("E", "P", "V", "S")

%saveas(gcf,'figures/4D_meanstd_noise.png')


%% Sanity check
% look how this works for variaous tau > when tau is really small and big

% small tau (4/1000 for E; 2/1000 for I) doesn't affect the ode too much, however changing it
% to a really big tau (10000/1000) affects all cells, with S least
% affected

% Didn't change tau noise (50/1000)


%% Rate output for h is 0, 2, 15
figure;
h_range = [0, 2, 15];
%u = zeros(4, length(t));
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(4,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii)* dt./tau; 
    end    
    
    %plot for every h the rates in one plot
    subplot(1, 3, m)
    plot(t, u) 
    ylabel("rate")
    xlabel("time")
    legend("E", "P", "V", "S")

end


%% Covariance



%% Parameter search
% Find the values for parameter setting (h_tot) in arousal/locomotion and attention
% h_tot = a * [1;b], a = h_range, b = I_range
a_range = 0:0.5:15;
%I_range = repmat((-3:0.2:3),3,1); %range for Is, but all similar range!
%E_range = ones(1,length(I_range));
%b_range = vertcat(E_range, I_range);
b_range=-3:0.2:3;
par_change = zeros(length(b_range),length(a_range), 4, length(t));
for b = 1:length(b_range)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range(a))
    
        % update I_range input    
         h = vertcat(1,repmat(b_range(b),3,1)) * a_range(a);
         fprintf('E input: %d\n', h(1))
         fprintf('I input: %d, %d, %d\n', h(2), h(3), h(4))

    
        %Generate noise vector
        for i = 1:length(t)-1
            eta(:,i+1) = eta(:,i) + (-eta(:,i) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(4,1))) *(1./tau_noise);
        end

        %Integrate neural system with noise forcing
         for ii = 1: length(eta)-1  
           % Forward Euler step + x(i) which is the noise
              u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; %set eta * 0 to remove noise
         end
     
        
          % add u to matrix 
            par_change(b,a,:,:) = u;
        
    end
end

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

save('data/4Drate_par_change.mat', 'par_change') %this one is not being saved
save('data/4Drate_mean_par.mat', 'mean_par')
save('data/4Drate_stds_par.mat', 'stds_par')

% plot stats
figure;
xticklabels = a_range(1:3:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
yticklabels = b_range(1:4:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));

subplot(2,4,1)
imagesc(mean_par(:,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,2)
imagesc(mean_par(:,:,2))
title("mean rate PV")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,3)
imagesc(mean_par(:,:,3))
title("mean rate VIP")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,4)
imagesc(mean_par(:,:,4))
title("mean rate SOM")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar


subplot(2,4,5)
imagesc(stds_par(:,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,6)
imagesc(stds_par(:,:,2))
title("std dev rate PV")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,7)
imagesc(stds_par(:,:,3))
title("std dev rate VIP")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar

subplot(2,4,8)
imagesc(stds_par(:,:,4))
title("std dev rate SOM")
xlabel("a-range")
ylabel("b-range")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
xtickangle(45)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
colorbar




%mesh plots
figure;
xticklabels = a_range(1:3:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
yticklabels = b_range(1:4:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));

subplot(2,4,1)
surf(mean_par(:,:,1), 'FaceAlpha',0.5)
title('mean E')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,2)
surf(mean_par(:,:,2), 'FaceAlpha',0.5)
title('mean PV')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,3)
surf(mean_par(:,:,3), 'FaceAlpha',0.5)
title('mean VIP')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,4)
surf(mean_par(:,:,4), 'FaceAlpha',0.5)
title('mean SOM')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,4,5)
surf(stds_par(:,:,1), 'FaceAlpha',0.5)
title('std dev E')
xlabel('a-range')
ylabel('b-range')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,6)
surf(stds_par(:,:,2), 'FaceAlpha',0.5)
title('std dev PV')
xlabel('a-range')
ylabel('b-range')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,7)
surf(stds_par(:,:,3), 'FaceAlpha',0.5)
title('std dev VIP')
xlabel('a-range')
ylabel('b-range')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,4,8)
surf(stds_par(:,:,4), 'FaceAlpha',0.5)
title('std dev SOM')
xlabel('a-range')
ylabel('b-range')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%% Identify transient => DISCUSS
%transient might affect results. Look at the time trace to identify and
%exclude transient

trans = squeeze(par_change(2,:,1,:)); %matrix for first input b and E cells

figure;
plot(t, trans) %zoom into very beginning
xlabel("time")
ylabel("u")

%seems that transient is before t = 0.1
find(t == 0.03) %idx = 101

par_change_trans(:,:,:,:) = par_change(:,:,:,101:end);

% mean and std
mean_par_trans = mean(par_change_trans, 4);
stds_par_trans= std(par_change_trans,0,4);




%% Cross correlogram h=2 & h=15
% create cross corr for b = 1, and a = 2 & 15 to resembles Hennequin fig1E

% a =2, E and I cells
idx_b = b_range == 1;
idx_a = a_range == 2;

dat_a2 = squeeze(par_change(idx_b,idx_a,:,:))';
dat_a2n = dat_a2 - mean(dat_a2,1); % Normalize: substract mean value of time trace

[corr_2,lags_2] = xcorr(dat_a2n, 'coeff');


% a =15, E and I cells
idx_a = a_range == 15;

dat_a15 = squeeze(par_change(idx_b,idx_a,:,:))';
dat_a15n = dat_a15 - mean(dat_a15,1); 

[corr_15,lags_15] = xcorr(dat_a15n, 'coeff');


titles = {'r_E - r_E','r_E - r_P','r_E - r_V','r_E - r_S', ...
    'r_P - r_E', 'r_P - r_P', 'r_P - r_V', 'r_P -r_S', ...
    'r_V - r_E', 'r_V - r_P', 'r_V - r_V', 'r_V - r_S', ...
    'r_S - r_E', 'r_S - r_P', 'r_S -r_V', 'r_S - r_S'};
figure;
for row = 1:4
    for col = 1:4
        nm = 4*(row-1)+col;
        subplot(4,4,nm)
        stem(lags_2,corr_2(:,nm),'.', 'LineStyle', 'none')
        hold on
        stem(lags_15, corr_15(:,nm), '.', 'LineStyle', 'none')
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('a=2', 'a=15')
    end
end



%% Integral cross-corr for a and b
%corr_ab = zeros((2*length(t))-1,16, length(b_range), length(a_range));
for b = 1:length(b_range)
    
    % update b_range input
    fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range)
        
        %create normalized data
        dat_a = squeeze(par_change(b,a,:,:))';
        dat_an = dat_a - mean(dat_a,1); % Normalize: substract mean value of time trace
        
        %get cross correlogram for lags between 200ms
        [corr_a,lags_ab] = xcorr(dat_an, 200, 'coeff');
        
        corr_ab(:,:,b,a) = corr_a;
    end
    
end


%Get integral
for i = 1:size(corr_ab,3)
    for j = 1:size(corr_ab,4)
        for k = 1:size(corr_ab,2)
            intg(:,k,i,j) = trapz(corr_ab(:,k,i,j));
        end
    end
end


figure;
titles = {'r_E - r_E','r_E - r_P','r_E - r_V','r_E - r_S', ...
    'r_P - r_E', 'r_P - r_P', 'r_P - r_V', 'r_P -r_S', ...
    'r_V - r_E', 'r_V - r_P', 'r_V - r_V', 'r_V - r_S', ...
    'r_S - r_E', 'r_S - r_P', 'r_S -r_V', 'r_S - r_S'};
xticklabels = a_range(1:4:end);
xticks = linspace(1, size(intg, 3), numel(xticklabels));
yticklabels = b_range(1:5:end);
yticks = linspace(1, size(intg, 4), numel(yticklabels));

% mesh plot
for row = 1:4
    for col = 1:4
        nm = 4*(row-1)+col;
        intg_ab = squeeze(intg(:,nm,:,:)); %get separate integrals per corr        
        subplot(4,4,nm)
        surf(intg_ab, 'FaceAlpha',0.5)
        title(titles{1,nm}) %titles is defined in cross corr h=2, h=15
        xlabel('a-range')
        ylabel('b-range')
        zlabel('integral')
        set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
        set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
    end
end


for row = 1:4
    for col = 1:4
        nm = 4*(row-1)+col;
        intg_ab = squeeze(intg(:,nm,:,:)); %get separate integrals per corr        
        subplot(4,4,nm)
        imagesc(intg_ab)
        colorbar
        title(titles{1,nm}) %titles is defined in cross corr h=2, h=15
        xlabel('a-range')
        ylabel('b-range')
        zlabel('integral')
        set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
        set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
    end
end






%% Recreate fig 7c of Kuchibotla
% create separate input to cells and look at change from baseline (h = 0)

h_range = [0; 15; 0; 15]; %E, P, V, S (passive state: E and V; active state: P and S)
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = h_range;

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau;  %first try without noise (eta * 0)
    end
end


figure;
plot(t, u, 'LineWidth',2)
ylabel("rate")
xlabel("time")
legend("E", "P", "V", "S")

% h = 0 for all: shows that all rates go from 1 to 0 within 0.1ms (E is a bit slower)

%saveas(gcf, 'passive_context.png')
%saveas(gcf, 'active_context.png')


%% Passive vs Active

% passive state is stimulus only. As discussed with Yashar, stimulus only
% is h input to E and PV cells, although h does not have to be similar.

% In Kuchibhotla, passive state is increase of activity by Exc and VIP cells 
% i.e. h = 15 for E and h = 15 for VIP cells

% % run script with h = 15 for E cells and h = 15 for VIP
u_passive = u;

%Active state in Kuchibhotla is described as increased activity of PV and SOM. Set
% h=15 for PV and SOM

% run script with h = 15 only for E cells
u_active = u;



%% Modulation index 

% values from Kuchibhotla fig 3.
E_pass = 67;
E_act = 33;
P_pass = 39;
P_act = 61;
V_pass = 78;
V_act = 22;
S_pass = 40;
S_act = 60;

val = [E_pass E_act;
P_pass P_act;
V_pass V_act;
S_pass S_act;];

mod = zeros(length(val),1);
for n = 1:length(val)
    mod(n) = (val(n,2) - val(n,1))/val(n,2);
end


%values obtained with simulation 
act_mean = mean(u_active, 2);
pas_mean = mean(u_passive, 2);

val_sim = [pas_mean act_mean];

mod_sim = zeros(length(val_sim),1);
for m = 1:length(val)
    mod_sim(m) = (val_sim(m,2) - val_sim(m,1))/val_sim(m,2);
end


