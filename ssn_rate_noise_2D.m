% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model with OU process
%                          (noise added per dt)


%% Parameters
k = 0.3; %scaling constant 
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

% Noise process is Ornstein-Uhlenbeck
tau_noise = 50/1000; 
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
ode_rate = @(t, u, h)  (-u + k.*ReLU(W *u + h).^b)./tau;


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

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

save('data/par_change-b-3-3.mat', 'par_change')
save('data/mean_par-b-3-3.mat', 'mean_par')
save('data/stds_par-b-3-3.mat', 'stds_par')

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

%% Check values
%CHECK: only plot input [1;1]
b_range(21) 

figure;
subplot(2,2,1)
imagesc(mean_par(21,:,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(mean_par(21,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(21,:,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(mean_par(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(mean_par(21,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(21,:,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(:,:,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(stds_par(21,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(21,:,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:2:end);
xticks = linspace(1, size(stds_par(:,:,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(21);
yticks = linspace(1, size(stds_par(21,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)




%CHECK: look at the [1,1] input over the a-range. This should give the same
%plots as before
b_range(21)

figure;
subplot(1,2,1)
plot(a_range,mean_par(21,:,1), 'LineWidth', 1.5);
hold on
plot(a_range,mean_par(21,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("mean rate")
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(a_range,stds_par(21,:,1), 'LineWidth', 1.5)
hold on
plot(a_range,stds_par(21,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("std dev. rate")
ylabel("rate")
xlabel("h")


%% Look for potential b-values that represent state
%ZOOM: only plot values for a_range =< 5 (look at stds)
a_range(11) 

figure;
subplot(2,2,1)
imagesc(mean_par(:,1:11,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(mean_par(:,1:11,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(:,1:11,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(mean_par(:,1:11,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(mean_par(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(:,1:11,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(stds_par(:,1:11,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(stds_par(:,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(:,1:11,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(stds_par(:,1:11,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(1:2:end);
yticks = linspace(1, size(stds_par(:,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)



%CHECK: look at individual inputs [...,...] to check with 2D plot
idx_b = 11;
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


%ZOOM: b-values -0.4:3 in zoomed in a-range
b_range(14) 

figure;
subplot(2,2,1)
imagesc(mean_par(14:24,1:11,1))
title("mean rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(mean_par(:,1:11,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(14:24);
yticks = linspace(1, size(mean_par(14:24,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,2)
imagesc(mean_par(14:24,1:11,2))
title("mean rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(mean_par(:,1:11,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(14:24);
yticks = linspace(1, size(mean_par(14:24,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,3)
imagesc(stds_par(14:24,1:11,1))
title("std dev rate E")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(stds_par(:,1:11,1), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(14:24);
yticks = linspace(1, size(stds_par(14:24,:,1), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


subplot(2,2,4)
imagesc(stds_par(14:24,1:11,2))
title("std dev rate I")
xlabel("a-range")
ylabel("b-range")
colorbar

xticklabels = a_range(1:11);
xticks = linspace(1, size(stds_par(:,1:11,2), 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = b_range(14:24);
yticks = linspace(1, size(stds_par(14:24,:,2), 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


%Plotting b-range zoom
% loop through b-range(14:24)
figure;
for n=1:length(b_range(14:24))
    
    idx = 14+ (n-1);
    
    %E only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,1), 'LineWidth', 1);
    hold on
    legend(strcat('b =', num2str(b_range(14:24)')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,1), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range(14:24)')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end

figure;
for n=1:length(b_range(14:24))
    
    idx = 14+ (n-1);

    %I only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range(14:24)')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range(14:24)')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end



%Plotting b-range 
%b-range 1 is
b_range(21)

%b-range 0, -0.2, -0.4 is idx 16, 15, 14
b_range(16)

num = [21,16,15,14];
figure;
for n=1:length(b_range([21,16,15,14]))
    
    idx = num(n);

    %E only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,1), 'LineWidth', 1);
    hold on
    legend(strcat('b =', num2str(b_range([21,16,15,14])')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,1), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range([21,16,15,14])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end

figure;
for n=1:length(b_range([21,16,15,14]))    
    idx = num(n);
    %I only
    subplot(1,2,1)
    plot(a_range(1:11),mean_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =',num2str(b_range([21,16,15,14])')))
    title("mean rate")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(a_range(1:11),stds_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(b_range([21,16,15,14])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel("a-range")
end

%% Parameter search CHECK
a_range = (0:1:20);
%b_range = (-3:0.2:3); %range for I cell parameter values
par_change_c = zeros(length(a_range), 2, length(t));
%for b = 1:length(b_range)
    
    % update h_range input
    %fprintf('\n b-range: %d\n\n', b_range(b))
    
    for a = 1:length(a_range) % look over range of I input
        
        fprintf('\n a-value: %d\n', a_range(a))
    
        % update I_range input    
         h = [1;1] * a_range(a);
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
            par_change_c(a,:,:) = u;
        
    end
    
%end

%stats
mean_par_c = mean(par_change_c, 3);
stds_par_c= std(par_change_c,0,3);


figure;
subplot(2,2,1)
imagesc(mean_par_c(:,1))
colorbar

subplot(2,2,2)
imagesc(mean_par_c(:,2))
colorbar

subplot(2,2,3)
imagesc(stds_par_c(:,1))
colorbar

subplot(2,2,4)
imagesc(stds_par_c(:,2))
colorbar


figure;
plot(a_range,mean_par_c(:,1))
hold on
plot(a_range,mean_par_c(:,2))
figure;
plot(a_range,stds_par_c(:,1))
hold on
plot(a_range,stds_par_c(:,2))


%% What I-range gives the min std-dev over all h-ranges (look for lowest NC)
%get mean std dev of I-range values over h_range
E_stds_b = mean(stds_par(:,:,1)); 

figure;
plot(E_stds_b, '-rs', 'LineWidth', 1.5)
hold on
ylabel('std dev')

%get mean rate 
E_rate_b = mean(mean_par(:,:,1));

plot(E_rate_b, 'g', 'LineWidth', 2)
hold on
xlabel('idx I-range')
ylabel('rate')
title('b-param for E population')
legend("std dev", "rate")

%look where distance between nc and rate is largest
E_diff = (E_rate_b(:) - E_stds_b(:));
plot(E_diff, '-bs')

[E_diff_max, E_diff_max_idx] = max(E_diff);
b_val = b_range(E_diff_max_idx); %b value where nc and rate distance is largest 


%% Get attention and arousal state
% plot mean and std dev of I and E population to check h_range * [1; b_val] (not yet: h_range * (a * [1;b_val]))

h_range = (0:1:20);
stds_range = zeros(2, length(h_range));
mean_range = zeros(2, length(h_range));
for a = 1:length(h_range)
    
    % update input
    h_factor = h_range(a);
    disp(h_factor)
    h = [1;b_val] * h_factor; 

    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
    end
    

    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Forward Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
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
patch([1.5 3 3 1.5],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9]);
patch([14 max(xlim) max(xlim) 14],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9]);
% The order of the "children" of the plot determines which one appears on top.
% I need to flip it here.
set(gca,'children',flipud(get(gca,'children')))
% figure labels
title({"mean rate", "h_{range} * [1; b_{val}]"})
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(h_range, stds_range, 'Linewidth', 1.5)
legend("E","I", 'AutoUpdate','off')
% Add a patch
patch([1.5 3 3 1.5],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9])
patch([14 max(xlim) max(xlim) 14],[min(ylim) min(ylim) max(ylim) max(ylim)], [0.9 0.9 0.9])
% The order of the "children" of the plot determines which one appears on top.
% I need to flip it here.
set(gca,'children',flipud(get(gca,'children')))
% figure labels
title({"std dev. rate", "h_{range} * [1; b_{val}]"})
ylabel("rate")
xlabel("h")

%saveas(gcf, [pwd '/figures/2Drate_meanstd_loc_03.png'])






%% Get states

% h_spont 
%is where the nc is highest, ncHigh, which is at h = 2
%constant is ncHigh_hE as in the vector the value for E is 1
ncHigh = ncHigh_hE;

% get value for I in vector
%I_spont = ncHigh_hI/ncHigh_hE;
%h_spont = ncHigh*[1;I_spont];
h_spont = 2 * [1;1];


%h_stim 
%is where nc is lowest, ncLow, which is at h = 15
%constant is ncLow_hE as in the vector the value for E is 1
ncLow = ncLow_hE;

%get value for I in vector
% I_stim = ncLow_hI/ncLow_hE;
% h_stim = ncLow * [1;I_stim];
h_stim = 15 * [1;1];


%h_att
%NC decreases more for h_stim, so less fluctuation in rates and evoked rates
%increase (h > 15)
alpha = 0.07; %should be smaller than lambda
h_att = alpha * [1;0.2]; %play around with I value


%h_loc
% NC decreases more for h_stim, so less fluctuation in rates and evoked rates
%increase (h > 15). This is a large effect compared to h_att
% lambda = 0.3; %0.3 - 0.5 gives decreasing nc and relatively low sc, but all nc are higher than normal
% h_loc = lambda * [1;0.2]; %play around with I value: [1; 0.2] seems best

lambda = 0.5; %0.5 - 0.7 gives stable or increasing rate
h_loc = lambda * [1;1.1]; 


%% Rate output for h is h_spont, h_stim, h_att, h_loc
figure;
h_range = [h_spont, h_stim, h_att, h_loc];
for m = 1:length(h_range)
    
    % update input
    h = h_range(:,m);
    disp(h)
    
    %Generate noise vector
    for ii = 1:length(t)-1
        eta(:,ii+1) = eta(:,ii) + (-eta(:,ii) *dt + sqrt(2 .*dt*tau_noise*sigma_a.^2).*(randn(2,1))) *(1./tau_noise);
    end
    
    %Integrate neural system with noise forcing
    for ii = 1: length(eta)-1  
      % Take the Euler step + x(i) which is the noise
      u(:,ii+1) = u(:,ii) + ode_rate(t, (u(:,ii)), h)*dt + eta(:,ii) * dt./tau; 
    end
    
    subplot(1, 4, m)
    plot(t, u, 'Linewidth',1) %order plots: h_spont, h_stim, h_att, h_loc
    ylim([0 100])
    ylabel("rate")
    xlabel("time")
    legend("E", "I")

end
