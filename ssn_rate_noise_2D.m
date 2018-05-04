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



%% Parameter search with 2D plot. 
% Find the values for parameter setting (h_tot) in arousal/locomotion and attention
% h_tot = a * [1;b], a = h_range, b = I_range

a_range = (0:0.5:15);
b_range = (-3:0.2:3); %range for I cell parameter values
par_change = zeros(length(b_range),length(a_range), 2, length(t));
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
        
    end
end

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

%save('data/par_change-b-3-3-Tn10.mat', 'par_change')
%save('data/mean_par-b-3-3-Tn10.mat', 'mean_par')
%save('data/stds_par-b-3-3-Tn10.mat', 'stds_par')

% plot stats
figure;
xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
yticklabels = b_range(1:3:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));

subplot(2,2,1)
imagesc(mean_par(:,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,2)
imagesc(mean_par(:,:,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,3)
imagesc(stds_par(:,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,4)
imagesc(stds_par(:,:,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

%saveas(gcf, '2Drate_meanstd.png')


%mesh plots
figure;
subplot(2,2,1)
surf(mean_par(:,:,1), 'FaceAlpha',0.5)
title('mean E')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(mean_par(:,:,2), 'FaceAlpha',0.5)
title('mean I')
xlabel('a-range')
ylabel('b-range')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(stds_par(:,:,1), 'FaceAlpha',0.5)
title('std dev E')
xlabel('a-range')
ylabel('b-range')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(stds_par(:,:,2), 'FaceAlpha',0.5)
title('std dev I')
xlabel('a-range')
ylabel('b-range')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

%saveas(gcf, '2Drate_meanstd_mesh.png')


%% Identify transient
%transient might affect results. Look at the time trace to identify and
%exclude transient

trans = squeeze(par_change(1,:,1,:)); %matrix for first input b and E cells

figure;
plot(t, trans) %zoom into very beginning
xlabel("time")
ylabel("u")

%seems that transient is before t = 0.1
find(t == 0.1) %idx = 101

par_change_trans(:,:,:,:) = par_change(:,:,:,101:end);

% mean and std
mean_par_trans = mean(par_change_trans, 4);
stds_par_trans= std(par_change_trans,0,4);


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

%save('data/par_change-b-n02-1.mat', 'par_change2')
%save('data/mean_par-b-n02-1.mat', 'mean_par2')
%save('data/stds_par-b-n02-1.mat', 'stds_par2')

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



%% Cross correlogram
% create cross corr for b = 1, and a = 2 & 15 to resembles Hennequin fig1E

% a =2, E and I cells
idx_b = b_range == 1;
idx_a = a_range == 2;

dat_a2 = squeeze(par_change_trans(idx_b,idx_a,:,:))';
dat_a2n = dat_a2 - mean(dat_a2,1); % Normalize: substract mean value of time trace

[corr_2,lags_2] = xcorr(dat_a2n, 'coeff');


% a =15, E and I cells
idx_a = a_range == 15;

dat_a15 = squeeze(par_change_trans(idx_b,idx_a,:,:))';
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


%% Noise correlation as integral under cross-corr
% Spike count correlation are proportional to the integral under the spike
% train cross-correlaogram (Cohen & Kohn, 2011)

% Restrict calculations to lags between -0.2 and 0.2 seconds (200 ms)
[corr_2,lags_2] = xcorr(dat_a2n,200,'coeff');
[corr_15,lags_15] = xcorr(dat_a15n,200,'coeff');

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


%% Integral cross-corr for a and b
%corr_ab = zeros((2*length(t))-1,4, length(b_range), length(a_range));
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


%get separate areas
intg_EE = squeeze((intg(:,1,:,:)));
intg_EI = squeeze((intg(:,2,:,:)));
intg_IE = squeeze((intg(:,3,:,:)));
intg_II = squeeze((intg(:,4,:,:)));

%plots for integral values over range a and b
figure;
xticklabels = a_range(1:4:end);
xticks = linspace(1, size(intg, 3), numel(xticklabels));
yticklabels = b_range(1:4:end);
yticks = linspace(1, size(intg, 4), numel(yticklabels));

subplot(2,2,1)
imagesc(intg_EE)
title('r_E - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
imagesc(intg_EI)
title('r_E - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
imagesc(intg_IE)
title('r_I - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
imagesc(intg_II)
title('r_I - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


% mesh plot for integral values over range a and b
figure;
subplot(2,2,1)
surf(intg_EE, 'FaceAlpha',0.5)
title('r_E - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(intg_EI, 'FaceAlpha',0.5)
title('r_E - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(intg_IE, 'FaceAlpha',0.5)
title('r_I - r_E')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(intg_II, 'FaceAlpha',0.5)
title('r_I - r_I')
xlabel('a-range')
ylabel('b-range')
zlabel('integral')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



% separate plot for a=2 different b (1.6, 0.2, 0, -1.4)
find(b_range == 1)

titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_ab,corr_ab(:,nm,21,5),'.','LineWidth', 2, 'LineStyle','none') %b=1
        hold on
        stem(lags_ab,corr_ab(:,nm,24,5),'.','LineWidth', 2, 'LineStyle','none') %b=1.6
        hold on
        stem(lags_ab,corr_ab(:,nm,17,5), '.','LineWidth', 2, 'LineStyle','none') %b=0.2
        hold on
        stem(lags_ab,corr_ab(:,nm,16,5), '.','LineWidth', 2, 'LineStyle','none') %b=0
        hold on
        stem(lags_ab,corr_ab(:,nm,9,5), '.','LineWidth', 2, 'LineStyle','none') %b=-1.4
        title(titles{1,nm})
        ylim([0 1])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=1.6', 'b=0.2', 'b=0', 'b=-1.4')
    end
end


% separate plot for a=15 different b (1.6, 0.2, 0, -1.4)
find(b_range == 1)

titles = {'r_E - r_E','r_E - r_I','r_I - r_E','r_I - r_I'};
figure;
for row = 1:2
    for col = 1:2
        nm = 2*(row-1)+col;
        subplot(2,2,nm)
        stem(lags_ab,corr_ab(:,nm,21,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=1
        hold on
        stem(lags_ab,corr_ab(:,nm,24,31),'.','LineWidth', 2, 'LineStyle','none') %b=1.6
        hold on
        stem(lags_ab,corr_ab(:,nm,17,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=0.2
        hold on
        stem(lags_ab,corr_ab(:,nm,16,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=0
        hold on
        stem(lags_ab,corr_ab(:,nm,9,31),'.', 'LineWidth', 2, 'LineStyle','none') %b=-1.4
        title(titles{1,nm})
        ylim([0 1])
        ylabel('corr.')
        xlabel('time lag (ms)')
        legend('b=1', 'b=1.6', 'b=0.2', 'b=0', 'b=-1.4')
    end
end



%% Zoom #3: Integral cross-corr for a 0:5
a_range3 = (0:0.2:5);
b_range = (-3:0.2:3); 
par_change3 = zeros(length(b_range),length(a_range3), 2, length(t));
for b = 1:length(b_range)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range3) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range3(a))
    
        % update I_range input    
         h = [1;b_range(b)] * a_range3(a);
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
            par_change3(b,a,:,:) = u;
        
    end
end


save('data/par_change-a-0-4.mat', 'par_change')
save('data/mean_par-a-0-4.mat', 'mean_par')
save('data/stds_par-a-0-4.mat', 'stds_par')


%cross-correlation
for b = 1:length(b_range)
    
    % update b_range input
    fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range3)
        
        %create normalized data
        dat_a3 = squeeze(par_change3(b,a,:,:))';
        dat_an3 = dat_a3 - mean(dat_a3,1); % Normalize: substract mean value of time trace
        
        %get cross correlogram for lags between 200ms
        [corr_a3,lags_ab3] = xcorr(dat_an3, 200, 'coeff');
        
        corr_ab3(:,:,b,a) = corr_a3;
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
yticklabels = b_range(1:4:end);
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
            stem(lags_ab,corr_2b(:,nm,i),'.', 'LineStyle','none')
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
            stem(lags_ab,corr_15b(:,nm,i),'.', 'LineStyle','none')
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

xticklabels = a_range(1:4:end);
xticks = linspace(1, size(FF_par_trans(:,:,1), 1), numel(xticklabels));
yticklabels = b_range(1:2:end);
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




















