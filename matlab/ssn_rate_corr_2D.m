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


%% Cross correlogram
% create cross corr for h_I and h_E= 2 & 15 to resembles Hennequin fig1E

% h =2, E and I cells
idx_E = E_range == 2;
idx_I = I_range == 2;

dat_h2 = squeeze(par_change_trans(idx_E,idx_I,:,:))';
dat_h2n = dat_h2 - mean(dat_h2,1); % Normalize: substract mean value of time trace

[corr_2,lags_2] = xcorr(dat_h2n, 'coeff');


% h =15, E and I cells
idx_E = E_range == 15;
idx_I = I_range == 15;

dat_h15 = squeeze(par_change_trans(idx_E,idx_I,:,:))';
dat_h15n = dat_h15 - mean(dat_h15,1); 

[corr_15,lags_15] = xcorr(dat_h15n, 'coeff');


titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_2,corr_2(:,nm),'.')
        hold on
        stem(lags_15, corr_15(:,nm), '.')
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('h_{tot}=2', 'h_{tot}=15')
    end
end

% saveas(gcf, 'figures/2Drate_cacross_h215.png')


%change tau noise and run it again to look for effect on cross corr
% saved under: par_change-b-3-3-Tn10.mat


%% Noise correlation as integral under cross-corr
% Spike count correlation are proportional to the integral under the spike
% train auto-correlaogram (Kanishiro, 2017)

% Restrict calculations to lags between -0.2 and 0.2 seconds (200 ms)
[corr_2,lags_2] = xcorr(dat_h2n,200,'coeff');
[corr_15,lags_15] = xcorr(dat_h15n,200,'coeff');

% In corr_n, column 1 = E/E; c2 = E/I; c3= I/E; c4 = I/I
% sum over corr values from lag -200 to 200

for i = 1:size(corr_2,2)
    intg_2(:,i) = trapz(corr_2(:,i)); %using trapezoidal  rule for definite integral
end

for i = 1:size(corr_15,2)
    intg_15(:,i) = trapz(corr_15(:,i)); %using trapezoidal  rule for definite integral
end

figure;
stem(intg_2, 'filled', 'g')
hold on
stem(intg_15, 'filled', 'r')
hold on

% Compare integration trend with half-way value (0.5) trend
% find 0.5
mid2 = corr_2(1:round(length(corr_2)/2),:); %restrict to halve of the distribution, otherwise you get two values for 0.5
[corr2_val, corr2_idx] = min(abs(mid2-0.5));
%get lag time
lag_mid2 = abs(lags_2(corr2_idx));

mid15 = corr_15(1:round(length(corr_15)/2),:); 
[corr15_val, corr15_idx] = min(abs(mid15-0.5));
%get lag time
lag_mid15 = abs(lags_15(corr15_idx));


stem(lag_mid2,'g')
hold on
stem(lag_mid15,'r')
legend('integral h=2', 'integral h =15', 'mid h=2', 'mid h=15')

%Result: seems to follow similar trend. I will continue with the integral
%of the cross correlograms


%% Integral cross-corr for E and I
%get cross and auto correlations
for e = 1:length(E_range)
    
    % update b_range input
    fprintf('\n E-range: %d\n\n', E_range(e))
    
    for i = 1:length(I_range)
        
        %create normalized data
        dat_h = squeeze(par_change_trans(e,i,:,:))';
        dat_hn = dat_h - mean(dat_h,1); % Normalize: substract mean value of time trace
        
        %get cross correlogram for lags between 200ms
        [corr_h,lags_h] = xcorr(dat_hn, 200, 'coeff');
        
        corr_htot(:,:,e,i) = corr_h;
    end
end


%Get integral
for i = 1:size(corr_htot,3)
    for j = 1:size(corr_htot,4)
        for k = 1:size(corr_htot,2)
            intg(:,k,i,j) = trapz(corr_htot(:,k,i,j));
        end
    end
end


%% Plots cross and auto correlations

%get separate areas
intg_EE = squeeze((intg(:,1,:,:)));
intg_EI = squeeze((intg(:,2,:,:)));
intg_IE = squeeze((intg(:,3,:,:)));
intg_II = squeeze((intg(:,4,:,:)));

%plots for integral values over range E and I
figure;
xticklabels = E_range(1:4:end);
xticks = linspace(1, size(intg, 4), numel(xticklabels));
yticklabels = I_range(1:4:end);
yticks = linspace(1, size(intg, 4), numel(yticklabels));

subplot(2,2,1)
imagesc(intg_EE)
title('r_E - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,2)
imagesc(intg_EI)
title('r_E - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,3)
imagesc(intg_IE)
title('r_I - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,4)
imagesc(intg_II)
title('r_I - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)

% saveas(gcf, 'figures/2Drate_cacross_twoInd0_15.png')

% mesh plot for integral values over range E and I
figure;
xticklabels = E_range(1:6:end);
xticks = linspace(1, size(intg, 4), numel(xticklabels));
yticklabels = I_range(1:6:end);
yticks = linspace(1, size(intg, 4), numel(yticklabels));

subplot(2,2,1)
surf(intg_EE, 'FaceAlpha',0.5)
title('r_E - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,2)
surf(intg_EI, 'FaceAlpha',0.5)
title('r_E - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,3)
surf(intg_IE, 'FaceAlpha',0.5)
title('r_I - r_E')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])
subplot(2,2,4)
surf(intg_II, 'FaceAlpha',0.5)
title('r_I - r_I')
xlabel('h_E', 'FontSize', 12)
ylabel('h_I', 'FontSize', 12)
zlabel('integral', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 100])

% saveas(gcf, 'figures/2Drate_cacross_twoInd0_15_mesh.png')


%% Correlation for specific h_tot
% separate plot for a=2 different b (1.6, 0.2, 0, -1.4)
find(I_range == 1)

titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_E,corr_ab(:,nm,21,5),'.','LineWidth', 2, 'LineStyle','none') %b=1
        hold on
        stem(lags_E,corr_ab(:,nm,24,5),'.','LineWidth', 2, 'LineStyle','none') %b=1.6
        hold on
        stem(lags_E,corr_ab(:,nm,17,5), '.','LineWidth', 2, 'LineStyle','none') %b=0.2
        hold on
        stem(lags_E,corr_ab(:,nm,16,5), '.','LineWidth', 2, 'LineStyle','none') %b=0
        hold on
        stem(lags_E,corr_ab(:,nm,9,5), '.','LineWidth', 2, 'LineStyle','none') %b=-1.4
        title(titles{1,nm})
        ylim([0 1])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=1.6', 'b=0.2', 'b=0', 'b=-1.4')
    end
end


% separate plot for a=15 different b (1.6, 0.2, 0, -1.4)
find(I_range == 1)

titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
f4 = figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_E,corr_ab(:,nm,21,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=1
        hold on
        stem(lags_E,corr_ab(:,nm,24,31),'.','LineWidth', 2, 'LineStyle','none') %b=1.6
        hold on
        stem(lags_E,corr_ab(:,nm,17,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=0.2
        hold on
        stem(lags_E,corr_ab(:,nm,16,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=0
        hold on
        stem(lags_E,corr_ab(:,nm,9,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=-1.4
        title(titles{1,nm})
        ylim([0 1])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=1.6', 'b=0.2', 'b=0', 'b=-1.4')
    end
end



%% Zoom #3: Integral cross-corr for a 0:5
a_range3 = (0:0.2:5);
I_range = (-3:0.2:3); 
par_change3 = zeros(length(I_range),length(a_range3), 2, length(t));
for i = 1:length(I_range)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', I_range(i))
    
    for e = 1:length(a_range3) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range3(e))
    
        % update I_range input    
         h = [1;I_range(i)] * a_range3(e);
         fprintf('E input: %d\n', h(1))
         fprintf('I input: %d\n', h(2))

    
        %Generate noise vector
        for ii = 1:length(t)-1
            eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
        end

        %Integrate neural system with noise forcing
        for ii = 1: length(eta)-1
            % Forward Euler step + x(i) which is the noise
            u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
        end
     
        
          % add u to matrix 
            par_change3(i,e,:,:) = u;
        
    end
end


save('data/par_change-a-0-4.mat', 'par_change')
save('data/mean_par-a-0-4.mat', 'mean_par')
save('data/stds_par-a-0-4.mat', 'stds_par')


%cross-correlation
for i = 1:length(I_range)
    
    % update b_range input
    fprintf('\n b-range: %d\n\n', I_range(i))
    
    for e = 1:length(a_range3)
        
        %create normalized data
        dat_a3 = squeeze(par_change3(i,e,:,:))';
        dat_an3 = dat_a3 - mean(dat_a3,1); % Normalize: substract mean value of time trace
        
        %get cross correlogram for lags between 200ms
        [corr_a3,lags_ab3] = xcorr(dat_an3, 200, 'coeff');
        
        corr_ab3(:,:,i,e) = corr_a3;
    end
    
end


%Get integral
for i = 1:size(corr_ab3,3)
    for j = 1:size(corr_ab3,4)
        for k = 1:size(corr_ab3,2)
            intg3(:,k,i,j) = trapz(corr_ab3(:,k,i,j));
        end
    end
end


%get separate areas
intg_EE3 = squeeze((intg3(:,1,:,:)));
intg_EI3 = squeeze((intg3(:,2,:,:)));
intg_IE3 = squeeze((intg3(:,3,:,:)));
intg_II3 = squeeze((intg3(:,4,:,:)));

% mesh plot for integral values over range a and b
figure;
xticklabels = a_range3(1:2:end);
xticks = linspace(1, size(intg3, 3), numel(xticklabels));
yticklabels = I_range(1:4:end);
yticks = linspace(1, size(intg3, 4), numel(yticklabels));

subplot(2,2,1)
surf(intg_EE3, 'FaceAlpha',0.5)
title('r_E - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(intg_EI3, 'FaceAlpha',0.5)
title('r_E - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(intg_IE3, 'FaceAlpha',0.5)
title('r_I - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(intg_II3, 'FaceAlpha',0.5)
title('r_I - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%% Plot selection b-values -0.6:0.2
% b=-0.6:0.2 and 1
num_c = [21,13,14,15,16,17];
%get corr values for b=-0.6:0.2 for a=2
corr_2b=corr_ab(:,:,num_c,5);
%get corr values for b=-0.6:0.2 for a=15
corr_15b=corr_ab(:,:,num_c,31);


titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        for i=1:size(corr_2b,3)
            stem(lags_E,corr_2b(:,nm,i),'.', 'LineStyle','none')
            hold on
            %stem(lags_ab, corr_15b(:,nm,i), '.', 'LineStyle','none')
        end
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=-0.6', 'b=-0.4', 'b=-0.2', 'b=0', 'b=0.2')
    end
end


titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        for i=1:size(corr_2b,3)
            stem(lags_E,corr_15b(:,nm,i),'.', 'LineStyle','none')
            hold on
        end
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=-0.6', 'b=-0.4', 'b=-0.2', 'b=0', 'b=0.2')
    end
end




%% Fano factors

%get variance (std dev^2)
var_par = stds_par.^2;
% FF = var / mean
FF_par = var_par./mean_par;

%get FF for data without transient
var_par_trans = stds_par_trans.^2;
FF_par_trans = var_par_trans./mean_par_trans;

%plot
figure;

xticklabels = E_range(1:4:end);
xticks = linspace(1, size(FF_par_trans(:,:,1), 1), numel(xticklabels));
yticklabels = I_range(1:2:end);
yticks = linspace(1, size(FF_par_trans(:,:,1), 2), numel(yticklabels));

subplot(2,1,1)
surf(FF_par_trans(:,:,1), 'FaceAlpha',0.5)
legend("E","I", 'AutoUpdate','off')
title("FF E")
xlabel("a-range")
ylabel("b-range")
zlabel("FF")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,1,2)
surf(FF_par_trans(:,:,2), 'FaceAlpha',0.5)
legend("E","I", 'AutoUpdate','off')
title("FF I")
xlabel("a-range")
ylabel("b-range")
zlabel("FF")
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)










