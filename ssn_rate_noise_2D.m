% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


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
dt = 1e-3;
t = 0:dt:100;

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


%% Generate a graph of fluctuations versus input (noise can be excluded; line 101)

h_range = (0:0.5:20);
%h_range = 0:1;         %to check for dynamics for no input
stds_range = zeros(2, length(h_range));
mean_range = zeros(2, length(h_range));
for a = 1:length(h_range)
    
    % update input
    h_factor = h_range(a);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    

    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
    end
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; %set eta * 0 to remove noise
    end
  
    
     % Get mean and std
    mean_range(:,a) = mean(u, 2);
    stds_range(:,a)= std(u,0,2);
    
end

figure;
subplot(1,2,1)
plot(h_range, mean_range, 'Linewidth', 1.5);
legend("E","I", 'AutoUpdate','off')
% Add a patch
%patch([1.5 3 3 1.5],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9]);
%patch([14 20 20 14],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9]);
% The order of the "children" of the plot determines which one appears on top.
% I need to flip it here.
%set(gca,'children',flipud(get(gca,'children')))
% figure labels
title("mean rate")
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(h_range, stds_range, 'Linewidth', 1.5)
legend("E","I", 'AutoUpdate','off')
% Add a patch
%patch([1.5 3 3 1.5],[min(ylim) min(ylim) 2 2], [0.9 0.9 0.9])
%patch([14 20 20 14],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9])
% The order of the "children" of the plot determines which one appears on top.
% I need to flip it here.
%set(gca,'children',flipud(get(gca,'children')))
% figure labels
title("std dev. rate")
ylabel("rate")
xlabel("h")



%% Rate output for h is 0, 2, 15
figure;
h_range = [0, 2, 15];
for m = 1:length(h_range)
    
    % update input
    h_factor = h_range(m);
    disp(h_factor)
    h = ones(2,1) * h_factor;
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    subplot(1, 3, m)
    plot(t, u, 'Linewidth',1)
    ylim([-2 50])
    ylabel("rate")
    xlabel("time")
    legend("E", "I")

end

%saveas(gcf, [pwd '/figures/2Drate_h0215.png'])


%% Get Noise Correlations (NC)
%High NC at what rate and h
[ncHigh_E, ncHigh_idxE]= max(stds_range(1,:));
[ncHigh_I, ncHigh_idxI] = max(stds_range(2,:));

ncHigh_hE = h_range(ncHigh_idxE);
ncHigh_hI = h_range(ncHigh_idxI);


% Smallest NC, starting at the high NC of matrix to avoid low NC between h=0-2
[ncLow_E, ncLow_idxE]= min(stds_range(1,ncHigh_idxE:end));
[ncLow_I, ncLow_idxI]= min(stds_range(2,ncHigh_idxI:end));

ncLow_hE = h_range(ncLow_idxE);
ncLow_hI = h_range(ncLow_idxI);



%% Parameter search with 2D plot. 
% Find the values for parameter setting (h_tot) in arousal/locomotion and attention
% h_tot = a * [1;b], a = h_range, b = I_range

a_range = (0:0.5:15);
b_range = (-3:0.2:3); %range for I cell parameter values
par_change = zeros(length(b_range),length(a_range), 2, length(t));
%stds_range_a = zeros(length(a_range),2);
%mean_range_a = zeros(length(a_range),2);
%par_change_mean = zeros(length(b_range),length(a_range),2);
%par_change_stds = zeros(length(b_range),length(a_range),2);
for b = 1:length(b_range)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range(a))
    
        % update I_range input    
         h = [1;b_range(b)] * a_range(a);
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
            par_change(b,a,:,:) = u;
        
          %stats
            %mean_range_a(a,:) = mean(u,2);
            %stds_range_a(a,:) = std(u, 0, 2); 
        
    end
    
            %par_change_mean(b,:,:) = mean_range_a;
            %par_change_stds(b,:,:) = stds_range_a;  
    
end

%par_change_noTrans(:,:,:,:) = par_change(:,:,:,101:end);

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

%save('data/par_change-b-3-3-Tn10.mat', 'par_change')
%save('data/mean_par-b-3-3-Tn10.mat', 'mean_par')
%save('data/stds_par-b-3-3-Tn10.mat', 'stds_par')

% plot stats
figure;
subplot(2,2,1)
imagesc(mean_par(:,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(:,:,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(mean_par(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(:,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(stds_par(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(:,:,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(stds_par(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%stats using 3D matrices for mean and stds CHECK
% figure;
% subplot(2,2,1)
% imagesc(par_change_mean(:,:,1))
% title("mean rate E")
% ylabel("a-range")
% xlabel("b-range")
% colorbar
% 
% xticklabels = b_range(1:3:end);
% xticks = linspace(1, size(par_change_mean(:,:,1), 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
% 
% yticklabels = a_range(1:2:end);
% yticks = linspace(1, size(par_change_mean(:,:,1), 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% 
% 
% subplot(2,2,2)
% imagesc(par_change_mean(:,:,2))
% title("mean rate I")
% ylabel("a-range")
% xlabel("b-range")
% colorbar
% 
% xticklabels = b_range(1:3:end);
% xticks = linspace(1, size(par_change_mean(:,:,2), 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
% 
% yticklabels = a_range(1:2:end);
% yticks = linspace(1, size(par_change_mean(:,:,2), 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% 
% 
% subplot(2,2,3)
% imagesc(par_change_stds(:,:,1))
% title("std dev E")
% ylabel("a-range")
% xlabel("b-range")
% colorbar
% 
% xticklabels = b_range(1:3:end);
% xticks = linspace(1, size(par_change_stds(:,:,1), 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
% 
% yticklabels = a_range(1:2:end);
% yticks = linspace(1, size(par_change_stds(:,:,1), 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% 
% 
% subplot(2,2,4)
% imagesc(par_change_stds(:,:,2))
% title("std dev I")
% ylabel("a-range")
% xlabel("b-range")
% colorbar
% 
% xticklabels = b_range(1:3:end);
% xticks = linspace(1, size(par_change_stds(:,:,2), 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
% 
% yticklabels = a_range(1:2:end);
% yticks = linspace(1, size(par_change_stds(:,:,2), 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% 

%% Check values for similar input
%CHECK: only plot input [1;1]
idx_b = find(b_range == 1);

figure;
subplot(2,2,1)
imagesc(mean_par(idx_b,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(idx_b,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(mean_par(idx_b,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(idx_b,:,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(idx_b,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(mean_par(idx_b,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(idx_b,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(idx_b,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(stds_par(idx_b,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(idx_b,:,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(idx_b,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(stds_par(idx_b,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%CHECK: look at the [1,1] input over the a-range. This should give the same
%plots as before
figure;
subplot(1,2,1)
plot(a_range,mean_par(idx_b,:,1), 'LineWidth', 1.5);
hold on
plot(a_range,mean_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("mean rate")
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(a_range,stds_par(idx_b,:,1), 'LineWidth', 1.5)
hold on
plot(a_range,stds_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("std dev. rate")
ylabel("rate")
xlabel("h")




%% Cross correlogram
% create cross corr for b = 1, and a = 2 & 15 to resembles Hennequin fig1E

% a =2, E and I cells
idx_b = b_range == 1;
idx_a = a_range == 2;

dat_a2 = squeeze(par_change(idx_b,idx_a,:,:))';
dat_a2n = dat_a2 - mean(dat_a2,1); %substract mean value of time trace

[corr_2,lags_2] = xcorr(dat_a2n, 'coeff');


% a =15, E and I cells
idx_a = a_range == 15;

dat_a15 = squeeze(par_change(idx_b,idx_a,:,:))';
dat_a15n = dat_a15 - mean(dat_a15,1); 

[corr_15,lags_15] = xcorr(dat_a15n, 'coeff');


titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_2,corr_2(:,nm),'.')
        hold on
        stem(lags_15, corr_15(:,nm), '.')
        %title(sprintf('E/I_{%d%d}',row,col))
        title(titles{1,nm})
        ylim([0 1])
        xlim([-200 200])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('a=2', 'a=15')
    end
end


%change tau noise and run it again to look for effect on cross corr
% saved under: par_change-b-3-3-Tn10.mat




% %Check: restrict calculations to lags between -0.2 and 0.2 seconds (200 ms)
% [corr_2,lags_2] = xcorr(dat_a2,200,'coeff');
% 
% figure;
% for row = 1:2
%     for col = 1:2
%         nm = 2*(row-1)+col;
%         subplot(2,2,nm)
%         stem(lags_2,corr_2(:,nm),'.')
%         title(sprintf('E/I_{%d%d}',row,col))
%         ylim([0 1])
%         xlim([-200 200])
%     end
% end


%% Indicate b-values that give a high std dev compared to b = 1
% look at the mean value for std dev in the a-range 10-15

%mean value for std dev for b=1 in a_range(10:15)
find(b_range == 1)
mean_std_1 = mean(stds_par(21,21:end,1));

%mean value for std dev for all bs in a_range(10:15)
mean_std = mean(stds_par(:,21:end,1),2);

figure;
subplot(1,2,1)
plot(b_range,mean_std)
line([1 1], [0 90], 'Color','red', 'LineWidth', 1.5); 
xlabel('b-range')
ylabel('mean std dev rate')
title('all b-values in a-range (10-15)')

%remove b-values that are 4 x b=1 value.
mean_std_sm = [mean_std(mean_std<3*mean_std_1), find(mean_std<3*mean_std_1)];

subplot(1,2,2)
plot(b_range(mean_std_sm(:,2)), mean_std_sm(:,1))
line([1 1], [0 0.6], 'Color','red', 'LineWidth', 1.5); 
xlabel('b-range')
ylabel('mean std dev rate')
title('selection b-range')


%% Look for b-value at the increased variability at h=2
%ZOOM: only plot values for a_range =< 5 (look at stds)
idx_a = find(a_range == 5);

figure;
subplot(2,2,1)
imagesc(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:11);
xticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(mean_std_sm(1,2):mean_std_sm(end,2));
yticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:11);
xticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(mean_std_sm(1,2):mean_std_sm(end,2));
yticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:11);
xticks = linspace(1, size(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(mean_std_sm(1,2):mean_std_sm(end,2));
yticks = linspace(1, size(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:11);
xticks = linspace(1, size(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(mean_std_sm(1,2):mean_std_sm(end,2));
yticks = linspace(1, size(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%CHECK: look at individual inputs [...,...] to check with 2D plot
idx_b = find(b_range == -1); %change what to find to look at different plots
b_range(idx_b)

figure;
subplot(1,2,1)
plot(a_range,mean_par(idx_b,:,1), 'LineWidth', 1.5);
hold on
plot(a_range,mean_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("mean rate")
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(a_range,stds_par(idx_b,:,1), 'LineWidth', 1.5)
hold on
plot(a_range,stds_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("std dev. rate")
ylabel("rate")
xlabel("h")


%% Plotting b-values 0.4, 0.6, 0.8 vs b = 1
%b-range 1 is
b_range(21)

%b-range 0.4, 0.6, 0.8 is idx 18, 19, 20
b_range(16)

num = [21,18,19,20];
figure;
for n=1:length(b_range([21,18,19,20]))
    
    idx = num(n);

    %E only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,1), 'LineWidth', 1);
    hold on
    legend(strcat('b =', num2str(b_range([21,18,19,20])')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,1), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range([21,18,19,20])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end

figure;
for n=1:length(b_range([21,18,19,20]))    
    idx = num(n);
    %I only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =',num2str(b_range([21,18,19,20])')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range([21,18,19,20])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end



%% ZOOM #2 (0.2:0.05:1)

a_range2 = (0:0.5:15);
b_range2 = (0.2:0.05:1); %range for I cell parameter values
par_change2 = zeros(length(b_range2),length(a_range2), 2, length(t));
for b = 1:length(b_range2)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', b_range2(b))
    
    for a = 1:length(a_range2) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range2(a))
    
        % update I_range input    
         h = [1;b_range2(b)] * a_range2(a);
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
            par_change2(b,a,:,:) = u;
        
    end
end

%stats using 4D matrix
mean_par2 = mean(par_change2, 4);
stds_par2= std(par_change2,0,4);

save('data/par_change-b-n02-1.mat', 'par_change')
save('data/mean_par-b-n02-1.mat', 'mean_par')
save('data/stds_par-b-n02-1.mat', 'stds_par')

% plot stats
figure;
subplot(2,2,1)
imagesc(mean_par2(:,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range2(1:2:end);
xticks = linspace(1, size(mean_par2(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range2(1:2:end);
yticks = linspace(1, size(mean_par2(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par2(:,:,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range2(1:2:end);
xticks = linspace(1, size(mean_par2(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range2(1:2:end);
yticks = linspace(1, size(mean_par2(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par2(:,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range2(1:2:end);
xticks = linspace(1, size(stds_par2(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range2(1:2:end);
yticks = linspace(1, size(stds_par2(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par2(:,:,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range2(1:2:end);
xticks = linspace(1, size(stds_par2(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range2(1:2:end);
yticks = linspace(1, size(stds_par2(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%mesh plots
figure;
xticklabels = a_range2(1:4:end);
xticks = linspace(1, size(mean_par2(:,:,1), 2), numel(xticklabels));
yticklabels = b_range2(1:2:end);
yticks = linspace(1, size(mean_par2(:,:,1), 1), numel(yticklabels));

subplot(2,2,1)
surf(mean_par2(:,:,1), 'FaceAlpha',0.5)
title('mean E')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(mean_par2(:,:,2), 'FaceAlpha',0.5)
title('mean I')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(stds_par2(:,:,1), 'FaceAlpha',0.5)
title('std dev E')
xlabel('a-range')
ylabel('b-range')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(stds_par2(:,:,2), 'FaceAlpha',0.5)
title('std dev I')
xlabel('a-range')
ylabel('b-range')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


%% Plot b = (0.55:0.7) vs b = 1

%b-range2 1 is
b_range2(17)

%b-range2 0.55, 0.6, 0.65, 0.7 is idx 8, 9, 10, 11
b_range2(8)

num = [17,8,9,10,11];

figure;
for n=1:length(num)
    
    idx = num(n);

    %E only
    subplot(1,2,1)
    plot(a_range2(:),mean_par2(idx,:,1), 'LineWidth', 1);
    hold on
    legend(strcat('b =', num2str(b_range2(num)')))
    title("mean rate E")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range2(:),stds_par2(idx,:,1), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range2(num)')))
    title("std dev. rate E")
    ylabel("rate")
    xlabel("a-range")
end


figure;
for n=1:length(num)    
    
    idx = num(n);
    
    %I only
    subplot(1,2,1)
    plot(a_range2(:),mean_par2(idx,:,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =',num2str(b_range2(num)')))
    title("mean rate I")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range2(:),stds_par2(idx,:,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range2(num)')))
    title("std dev. rate I")
    ylabel("rate")
    xlabel("a-range")
end










