% author:               Jantine Broek
% collaborator:         Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:                stabilized supralinear network model with OU process
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
dt = 1e-3;                                  % in sec
t = 0:dt:100;                             % until 100 sec

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


%% Functions

% ODE with rate as dynamical variable
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^n)./tau;

%% Data
% data from ssn_rate_noise_2D.m
cd(fullfile(dir_base, dir_data));

load('data/par_change-twoInd-0-15.mat')
load('data/mean_par-twoInd-0-15.mat')
load('data/stds_par-twoInd-0-15.mat')
load('data/h_range-twoInd-0-15.mat')


E_range = (0:0.5:15);
I_range = (0:0.5:15); 

%% Remove transient
%remove transient
par_change_trans(:,:,:,:) = par_change(:,:,:,101:end);

% mean and std
mean_par_trans = mean(par_change_trans, 4);
stds_par_trans= std(par_change_trans,0,4);


%% Covariance matrix for equal input
% create cov matrix for h_I and h_E= 2 & 15 to resembles Hennequin fig1E

% h =2, E and I cells
idx_E = E_range == 2;
idx_I = I_range == 2;

dat_h2 = squeeze(par_change_trans(idx_E,idx_I,:,:))';
dat_h2n = dat_h2 - mean(dat_h2,1); % Normalize: substract mean value of time trace

[cv_2,cv_lag2] = xcorr(dat_h2n, 'coeff');


% h =15, E and I cells
idx_E = E_range == 15;
idx_I = I_range == 15;

dat_h15 = squeeze(par_change_trans(idx_E,idx_I,:,:))';
dat_h15n = dat_h15 - mean(dat_h15,1); 

[cv_15,cv_lag15] = xcorr(dat_h15n, 'coeff');


titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(cv_lag2,cv_2(:,nm),'.')
        hold on
        stem(cv_lag15, cv_15(:,nm), '.')
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('cov.')
        xlabel('time lag (ms)')
        legend('h_{tot}=2', 'h_{tot}=15')
    end
end

cd(fullfile(dir_base, dir_fig));
saveas(gcf, '2Drate_cacov_h215.png')


%% Get covariance matrix for sweeps over E and I input

%get cross and auto covariances
for e = 1:length(E_range)
    
    % update b_range input
    fprintf('\n E-range: %d\n\n', E_range(e))
    
    for i = 1:length(I_range)
        
        %create normalized data
        dat_h = squeeze(par_change_trans(e,i,:,:))';
        dat_hn = dat_h - mean(dat_h,1); % Normalize: substract mean value of time trace
        
        %get cross correlogram for lags between 200ms
        [cv, cv_lag] = xcov(dat_hn, 200, 'coeff');
        
        %add all covs for h_tot
        cv_EI(:,:,i,e) = cv;
        
    end
end

%% Plot for specific h_tot

% plot for one cov ADJUST TO SPECIFIC ONE
titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for a = 1:2
    for b = 1:2
        nm = 2*(a-1)+b;
        subplot(2,2,nm)
        plot(cv_lag,cv_EI(:,nm))
        title(titles{1,nm})
        ylabel('cov.')
        xlabel('time lag (ms)')
        axis([-150 150 -0.2 1])
    end
end



%% Integral of covariance
%Get integral
for i = 1:size(cv_EI,3)
    for j = 1:size(cv_EI,4)
        for k = 1:size(cv_EI,2)
            intg_cv(:,k,i,j) = trapz(cv_EI(:,k,i,j));
        end
    end
end


%% Plots cross and auto covariance

%get separate areas
intg_cvEE = squeeze((intg_cv(:,1,:,:)));
intg_cvEI = squeeze((intg_cv(:,2,:,:)));
intg_cvIE = squeeze((intg_cv(:,3,:,:)));
intg_cvII = squeeze((intg_cv(:,4,:,:)));

%plots for integral values over range E and I
figure;
xticklabels = E_range(1:4:end);
xticks = linspace(1, size(intg_cv, 4), numel(xticklabels));
yticklabels = I_range(1:4:end);
yticks = linspace(1, size(intg_cv, 4), numel(yticklabels));

subplot(2,2,1)
imagesc(intg_cvEE)
title('r_E - r_E')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,2)
imagesc(intg_cvEI)
title('r_E - r_I')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,3)
imagesc(intg_cvIE)
title('r_I - r_E')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,4)
imagesc(intg_cvII)
title('r_I - r_I')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)

% saveas(gcf, 'figures/2Drate_cacov_twoInd0_15.png')

% mesh plot for integral values over range E and I
figure;
xticklabels = E_range(1:6:end);
xticks = linspace(1, size(intg_cv, 4), numel(xticklabels));
yticklabels = I_range(1:6:end);
yticks = linspace(1, size(intg_cv, 4), numel(yticklabels));

subplot(2,2,1)
surf(intg_cvEE, 'FaceAlpha',0.5)
title('r_E - r_E')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,2)
surf(intg_cvEI, 'FaceAlpha',0.5)
title('r_E - r_I')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,3)
surf(intg_cvIE, 'FaceAlpha',0.5)
title('r_I - r_E')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,4)
surf(intg_cvII, 'FaceAlpha',0.5)
title('r_I - r_I')
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])

% saveas(gcf, 'figures/2Drate_cacov_twoInd0_15_mesh.png')


%% Get noise correlation
% NC between neurons E and I is a normalized measure 
% p_{ij} = Cov(n_i, n_j)/sqrt(Var(n_i) Var(n_j))

p_EI = cv_EI(:,2,:,:) ./ sqrt(cv_EI(:,1,:,:) .* cv_EI(:,3,:,:));

nc = real(squeeze((p_EI(:,1,:,:))));

%plots for integral values over range E and I
figure;
xticklabels = E_range(1:4:end);
xticks = linspace(1, size(intg_cv, 4), numel(xticklabels));
yticklabels = I_range(1:4:end);
yticks = linspace(1, size(intg_cv, 4), numel(yticklabels));

subplot(2,2,1)
imagesc(intg_cvEE)
title('r_E - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,2)
imagesc(intg_cvEI)
title('r_E - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,3)
imagesc(intg_cvIE)
title('r_I - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,4)
imagesc(intg_cvII)
title('r_I - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)

% saveas(gcf, 'figures/2Drate_cacov_twoInd0_15.png')

% mesh plot for integral values over range E and I
figure;
xticklabels = E_range(1:6:end);
xticks = linspace(1, size(intg_cv, 4), numel(xticklabels));
yticklabels = I_range(1:6:end);
yticks = linspace(1, size(intg_cv, 4), numel(yticklabels));

subplot(2,2,1)
surf(intg_cvEE, 'FaceAlpha',0.5)
title('r_E - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,2)
surf(intg_cvEI, 'FaceAlpha',0.5)
title('r_E - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,3)
surf(intg_cvIE, 'FaceAlpha',0.5)
title('r_I - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,4)
surf(intg_cvII, 'FaceAlpha',0.5)
title('r_I - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])

% saveas(gcf, 'figures/2Drate_cacov_twoInd0_15_mesh.png')








