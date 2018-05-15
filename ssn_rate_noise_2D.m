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



%% Parameter search with one independent parameter
% Find the values for parameter setting (h_tot) in arousal/locomotion and attention
% h_tot = a * [1;b], a = h_range, b = I_range

E_range = (0:0.5:15);
I_range = (-3:0.2:3); %range for I cell parameter values
%par_change = zeros(length(I_range),length(E_range), 2, length(t));
tic
for e = 1:length(E_range)
    
    % update h_range input
    fprintf('\n E-range: %d\n\n', E_range(e))
    
    for i = 1:length(I_range) % look over range of I input
        
        fprintf('\n I-value: %d\n', I_range(i))
    
        % update I_range input    
        h = [1;I_range(i)] * E_range(e);
        % h = [E_range(e); I_range(i)];
         fprintf('E input: %d\n', h(1))
         fprintf('I input: %d\n', h(2))
         
         h_range(e,i,:) = h;

    
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
            par_change(e,i,:,:) = u;
        
    end
end
toc

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

%save('data/par_change-b-8-8.mat', 'par_change')
%save('data/mean_par-b-8-8.mat', 'mean_par')
%save('data/stds_par-b-8-8.mat', 'stds_par')
% save('data/h_range-b-8-8.mat', 'h_range')

%% Plots parameter search
% for [a;b]
% Ee_range = squeeze(h_range(1,:,:));
% Ee_range = Ee_range(1,:);
% Ii_range = squeeze(h_range(2,:,:))';
% Ii_range = Ii_range(1,:);

figure;
% for a*[1;b]
E_range = (0:0.5:15);
I_range = (-3:0.2:3); 
xticklabels = E_range(1:4:end);
xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
yticklabels = I_range(1:3:end);
yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));

% for [a;b]
% xticklabels = Ii_range(1:3:end);
% xticks = linspace(1, size(mean_par(:,:,1), 2), numel(xticklabels));
% yticklabels = Ee_range(1:3:end);
% yticks = linspace(1, size(mean_par(:,:,1), 1), numel(yticklabels));

% plot stats
subplot(2,2,1)
imagesc(mean_par(:,:,1))
title("mean rate E")
xlabel('h_I')
ylabel('h_E')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,2)
imagesc(mean_par(:,:,2))
title("mean rate I")
xlabel('h_I')
ylabel('h_E')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,3)
imagesc(stds_par(:,:,1))
title("std dev rate E")
xlabel('h_I')
ylabel('h_E')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,4)
imagesc(stds_par(:,:,2))
title("std dev rate I")
xlabel('h_I')
ylabel('h_E')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

%saveas(gcf, 'figures/2Drate_meanstd_b-8-8.png')


%mesh plots
figure;
subplot(2,2,1)
surf(mean_par(:,:,1), 'FaceAlpha',0.5)
title('mean E')
xlabel('h_I')
ylabel('h_E')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(mean_par(:,:,2), 'FaceAlpha',0.5)
title('mean I')
xlabel('h_I')
ylabel('h_E')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(stds_par(:,:,1), 'FaceAlpha',0.5)
title('std dev E')
xlabel('h_I')
ylabel('h_E')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(stds_par(:,:,2), 'FaceAlpha',0.5)
title('std dev I')
xlabel('h_I')
ylabel('h_E')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

%saveas(gcf, 'figures/2Drate_meanstd_mesh_b-8-8.png')



%% Plot histogram
% plot h_I input. This should spread out
plot(h_range(:,:,2))

mean_parE = mean_par(:,:,1);
stds_parE = stds_par(:,:,1);

figure;
subplot(2,1,1)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = mean_parE';
surf(x,y, C);
title('mean rate E')
xlabel('hE')
ylabel('hI')
colorbar

subplot(2,1,2)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = stds_parE';
surf(x,y, C);
title('std dev E')
xlabel('hE')
ylabel('hI')
colorbar


%% Identify transient
%transient might affect results. Look at the time trace to identify and
%exclude transient

trans = squeeze(par_change(7,7,1,:)); %matrix for first input b and E cells

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


%% Plots without transient
mean_parE_trans = mean_par_trans(:,:,1);
stds_parE_trans = stds_par_trans(:,:,1);

%surf
figure;
subplot(2,1,1)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = mean_parE_trans';
surf(x,y, C);
title('mean rate E')
xlabel('hE')
ylabel('hI')
colorbar

subplot(2,1,2)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = stds_parE_trans';
surf(x,y, C);
set(gca,'zscale','log');
title('std dev E')
xlabel('hE')
ylabel('hI')
colorbar


%pcolor
figure;
subplot(2,1,1)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = mean_parE_trans';
pcolor(x,y, C);
hold on
contour(C, [1 1],'color','r','lineWidth',1) %highlight [1;1] input
title('mean rate E')
xlabel('hE')
ylabel('hI')
colorbar

subplot(2,1,2)
x = h_range(:,:,1); %E input 0:15
y = h_range(:,:,2); %I input (0:15)*(-3:3) 
C = stds_parE_trans';
pcolor(x,y, C);
hold on
contour(C, [1.0 1.0],'color','r','lineWidth',1) %highlight [1;1] input
title('std dev E')
xlabel('hE')
ylabel('hI')
colorbar



%% Parameter search with one independent var and added spontaneous [2;2]
% h_tot = a * [1;b], a = h_range, b = I_range
clear h h_range par_change mean_par stds_par

E_range = (0:0.5:15);
I_range = (-3:0.2:3); %range for I cell parameter values
%par_change = zeros(length(I_range),length(E_range), 2, length(t));
tic
for e = 1:length(E_range)
    
    % update h_range input
    fprintf('\n E-range: %d\n\n', E_range(e))
    
    for i = 1:length(I_range) % look over range of I input
        
        fprintf('\n I-value: %d\n', I_range(i))
    
        % update I_range input    
        h = ([1;I_range(i)] * E_range(e)) + [2;2];
        % h = [E_range(e); I_range(i)];
         fprintf('E input: %d\n', h(1))
         fprintf('I input: %d\n', h(2))
         
         h_range(e,i,:) = h;

    
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
            par_change(e,i,:,:) = u;
        
    end
end
toc

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

save('data/par_change-spon.mat', 'par_change')
save('data/mean_par-spon.mat', 'mean_par')
save('data/stds_par-spon.mat', 'stds_par')
save('data/h_range-spon.mat', 'h_range')

%remove transient
par_change_trans(:,:,:,:) = par_change(:,:,:,101:end);

% mean and std
mean_par_trans = mean(par_change_trans, 4);
stds_par_trans= std(par_change_trans,0,4);

%% PLOTS
%surf
figure;
subplot(2,1,1)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = mean_par_trans(:,:,1);
surf(x,y, C);
title('mean rate E')
xlabel('hE')
ylabel('hI')
colorbar

subplot(2,1,2)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = stds_par_trans(:,:,1);
surf(x,y, C);
set(gca,'zscale','log');
title('std dev E')
xlabel('hE')
ylabel('hI')
colorbar

saveas(gcf, 'figures/2Drate_1indepvar_b-3-3_surf.png')

%pcolor
figure;
subplot(2,1,1)
x = h_range(:,:,1);
y = h_range(:,:,2);
C = mean_par_trans(:,:,1);
pcolor(x,y, C);
%hold on
%contour(C, [1 1],'color','r','lineWidth',1) %highlight [1;1] input
title('mean rate E')
xlabel('hE')
ylabel('hI')
colorbar

subplot(2,1,2)
x = h_range(:,:,1); %E input 0:15
y = h_range(:,:,2); %I input (0:15)*(-3:3) 
C = stds_par_trans(:,:,1);
pcolor(x,y, C);
%hold on
%contour(C, [1.0 1.0],'color','r','lineWidth',1) %highlight [1;1] input
title('std dev E')
xlabel('hE')
ylabel('hI')
colorbar

saveas(gcf, 'figures/2Drate_1indepvar_b-3-3_contour.png')


%% parameter search with 2 independent ranges

clear h h_range par_change mean_par stds_par

E_range = (0:0.5:15);
I_range = (0:0.5:15); 
%par_change = zeros(length(I_range),length(E_range), 2, length(t));
tic
for e = 1:length(E_range)
    
    % update h_range input
    fprintf('\n E-range: %d\n\n', E_range(e))
    
    for i = 1:length(I_range) % look over range of I input
        
        fprintf('\n I-value: %d\n', I_range(i))
    
        % update I_range input    
         h = [E_range(e);I_range(i)];
         fprintf('E input: %d\n', h(1))
         fprintf('I input: %d\n', h(2))
         
         h_range(e,i,:) = h;

    
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
          par_change(e,i,:,:) = u;
        
    end
end
toc

%stats using 4D matrix
mean_par = mean(par_change, 4);
stds_par= std(par_change,0,4);

% save('data/par_change-twoInd-0-15.mat', 'par_change', '-v7.3')
% save('data/mean_par-twoInd-0-15.mat', 'mean_par')
% save('data/stds_par-twoInd-0-15.mat', 'stds_par')
% save('data/h_range-twoInd-0-15.mat', 'h_range')

%remove transient
par_change_trans(:,:,:,:) = par_change(:,:,:,101:end);

% mean and std
mean_par_trans = mean(par_change_trans, 4);
stds_par_trans= std(par_change_trans,0,4);


%% Plots parameter search without transient
Ee_range = squeeze(h_range(:,1,:));
Ee_range = Ee_range(:,1);
Ii_range = squeeze(h_range(2,:,:))';
Ii_range = Ii_range(2,:);


% plot stats
figure;
xticklabels = Ii_range(1:6:end);
xticks = linspace(1, size(mean_par_trans(:,:,1), 2), numel(xticklabels));
yticklabels = Ee_range(1:6:end);
yticks = linspace(1, size(mean_par_trans(:,:,1), 1), numel(yticklabels));

subplot(2,2,1)
% imagesc(log(ReLU(mean_par_trans(:,:,1))))
imagesc(mean_par_trans(:,:,1))
title("mean rate E",'FontSize', 14)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,2)
% imagesc(log(ReLU(mean_par_trans(:,:,2))))
imagesc(mean_par_trans(:,:,2))
title("mean rate I", 'FontSize', 14)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,3)
% imagesc(log(ReLU(stds_par_trans(:,:,1))))
imagesc(stds_par_trans(:,:,1))
title("std dev rate E", 'FontSize', 14)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)
subplot(2,2,4)
% imagesc(log(ReLU(stds_par_trans(:,:,2))))
imagesc(stds_par_trans(:,:,2))
title("std dev rate I", 'FontSize', 14)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',14)

%saveas(gcf, 'figures/2Drate_meanstd_I-0-15_trans.pdf')


%mesh plots
figure;
subplot(2,2,1)
surf(mean_par_trans(:,:,1), 'FaceAlpha',0.5)
title('mean E', 'FontSize', 16)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('mean rate', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
subplot(2,2,2)
surf(mean_par_trans(:,:,2), 'FaceAlpha',0.5)
title('mean I', 'FontSize', 16)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('mean rate', 'FontSize', 12)
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
subplot(2,2,3)
surf(stds_par_trans(:,:,1), 'FaceAlpha',0.5)
title('std dev E', 'FontSize', 16)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('std dev', 'FontSize', 12)
%set(gca,'zscale','log');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 0.7])
subplot(2,2,4)
surf(stds_par_trans(:,:,2), 'FaceAlpha',0.5)
title('std dev I', 'FontSize', 16)
xlabel('h_I', 'FontSize', 12)
ylabel('h_E', 'FontSize', 12)
zlabel('st dev', 'FontSize', 12)
%set(gca,'zscale','log');
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
set(gca,'fontsize',13)
caxis([0 0.7])

%saveas(gcf, 'figures/2Drate_meanstd_mesh_I-0-15_trans.png')


%% Check for similar input [1;1]

for i = 1:size(h_range,1)
    meanE(:,i) = mean_par_trans(i,i,1);
    stdE(:,i) = stds_par_trans(i,i,1);
    hE(:,i) = h_range(i,1,1);
    
    for ii = 1:size(h_range,2)
        meanI(:,ii) = mean_par_trans(ii,ii,2);
        stdI(:,ii) = stds_par_trans(ii,ii,2);
        hI(:,ii) = h_range(1,ii,2);
    end 
end

figure;
subplot(2,1,1)
plot(meanE, 'LineWidth', 2)
hold on
plot(meanI,  'LineWidth', 2)
title('mean rate E')
xlabel('h_{tot}')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
subplot(2,1,2)
plot(stdE, 'LineWidth', 2)
hold on
plot(stdI, 'LineWidth', 2)
title('std dev E')
xlabel('h_{tot}')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

%saveas(gca, 'figures/2Drate_2indpvar_check_simh.png')


%% Look at vector space that indicate state vectors
% see below




%% Indicate b-values that give a high std dev compared to b = 1
% look at the mean value for std dev in the a-range 10-15

%mean value for std dev for b=1 in a_range(10:15)
find(I_range == 1)
mean_std_1 = mean(stds_par(21,21:end,1));

%mean value for std dev for all bs in a_range(10:15)
mean_std = mean(stds_par(:,21:end,1),2);

figure;
subplot(1,2,1)
plot(I_range,mean_std)
line([1 1], [0 90], 'Color','red', 'LineWidth', 1.5); 
xlabel('h_I')
ylabel('mean std dev rate')
title('all b-values in a-range (10-15)')

%remove b-values that are 4 x b=1 value.
mean_std_sm = [mean_std(mean_std<3*mean_std_1), find(mean_std<3*mean_std_1)];

subplot(1,2,2)
plot(I_range(mean_std_sm(:,2)), mean_std_sm(:,1))
line([1 1], [0 0.6], 'Color','red', 'LineWidth', 1.5); 
xlabel('h_I')
ylabel('mean std dev rate')
title('selection b-range')


%% Look for b-value at the increased variability at h=2
%ZOOM: only plot values for a_range =< 5 (look at stds)
idx_a = find(E_range == 5);

figure;
xticklabels = E_range(1:2:11);
xticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 2), numel(xticklabels));
yticklabels = I_range(mean_std_sm(1,2):mean_std_sm(end,2));
yticks = linspace(1, size(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1), 1), numel(yticklabels));

subplot(2,2,1)
imagesc(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1))
title("mean rate E")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,2)
imagesc(mean_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2))
title("mean rate I")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,3)
imagesc(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,1))
title("std dev rate E")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,4)
imagesc(stds_par(mean_std_sm(1,2):mean_std_sm(end,2),1:idx_a,2))
title("std dev rate I")
xlabel("E-range")
ylabel("I-range")
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


%CHECK: look at individual inputs [...,...] to check with 2D plot
idx_b = find(I_range == -1); %change what to find to look at different plots
I_range(idx_b)

figure;
subplot(1,2,1)
plot(E_range,mean_par(idx_b,:,1), 'LineWidth', 1.5);
hold on
plot(E_range,mean_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("mean rate")
ylabel("rate")
xlabel("h")


subplot(1,2,2)
plot(E_range,stds_par(idx_b,:,1), 'LineWidth', 1.5)
hold on
plot(E_range,stds_par(idx_b,:,2), 'LineWidth', 1.5)
legend("E","I", 'AutoUpdate','off')
title("std dev. rate")
ylabel("rate")
xlabel("h")


%% Plotting b-values 0.4, 0.6, 0.8 vs b = 1
%b-range 1 is
I_range(21)

%b-range 0.4, 0.6, 0.8 is idx 18, 19, 20
I_range(16)

num = [21,18,19,20];
figure;
for n=1:length(I_range([21,18,19,20]))
    
    idx = num(n);

    %E only
    subplot(1,2,1)
    plot(E_range(1:11),mean_par(idx,1:11,1), 'LineWidth', 1);
    hold on
    legend(strcat('h_I =', num2str(I_range([21,18,19,20])')))
    title("mean rate")
    ylabel("rate")
    xlabel('h_E')

    subplot(1,2,2)
    plot(E_range(1:11),stds_par(idx,1:11,1), 'LineWidth', 1)
    hold on
    legend(strcat('h_I =', num2str(I_range([21,18,19,20])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel('h_E')
end

figure;
for n=1:length(I_range([21,18,19,20]))    
    idx = num(n);
    %I only
    subplot(1,2,1)
    plot(E_range(1:11),mean_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('h_I =',num2str(I_range([21,18,19,20])')))
    title("mean rate")
    ylabel("rate")
    xlabel('h_E')

    subplot(1,2,2)
    plot(E_range(1:11),stds_par(idx,1:11,2), 'LineWidth', 1)
    hold on
    legend(strcat('h_I =', num2str(I_range([21,18,19,20])')))
    title("std dev. rate")
    ylabel("rate")
    xlabel('h_E')
end


%% ZOOM #2 (0.2:0.05:1)

E_range2 = (0:0.5:15);
I_range2 = (0.2:0.05:1); %range for I cell parameter values
par_change2 = zeros(length(I_range2),length(E_range2), 2, length(t));
for b = 1:length(I_range2)
    
    % update h_range input
    fprintf('\n b-range: %d\n\n', I_range2(b))
    
    for a = 1:length(E_range2) % look over range of I input
        
        fprintf('\n a-value: %d\n', E_range2(a))
    
        % update I_range input    
         h = [1;I_range2(b)] * E_range2(a);
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

save('data/par_change-b-n02-1.mat', 'par_change2')
save('data/mean_par-b-n02-1.mat', 'mean_par2')
save('data/stds_par-b-n02-1.mat', 'stds_par2')


%% Plot ZOOM #2 (0.2:0.05:1)
E_range2 = (0:0.5:15);
I_range2 = (0.2:0.05:1);

% plot stats
figure;
xticklabels = E_range2(1:4:end);
xticks = linspace(1, size(mean_par2(:,:,1), 2), numel(xticklabels));
yticklabels = I_range2(1:4:end);
yticks = linspace(1, size(mean_par2(:,:,1), 1), numel(yticklabels));

subplot(2,2,1)
imagesc(mean_par2(:,:,1))
title("mean rate E")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,2)
imagesc(mean_par2(:,:,2))
title("mean rate I")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,3)
imagesc(stds_par2(:,:,1))
title("std dev rate E")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

subplot(2,2,4)
imagesc(stds_par2(:,:,2))
title("std dev rate I")
xlabel('h_E')
ylabel('h_I')
colorbar
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

saveas(gcf, '2Drate_meanstd_02-1.png')


%mesh plots
figure;
xticklabels = E_range2(1:4:end);
xticks = linspace(1, size(mean_par2(:,:,1), 2), numel(xticklabels));
yticklabels = I_range2(1:4:end);
yticks = linspace(1, size(mean_par2(:,:,1), 1), numel(yticklabels));

subplot(2,2,1)
surf(mean_par2(:,:,1), 'FaceAlpha',0.5)
title('mean E')
xlabel('h_E')
ylabel('h_I')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,2)
surf(mean_par2(:,:,2), 'FaceAlpha',0.5)
title('mean I')
xlabel('h_E')
ylabel('h_I')
zlabel('mean rate')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,3)
surf(stds_par2(:,:,1), 'FaceAlpha',0.5)
title('std dev E')
xlabel('h_E')
ylabel('h_I')
zlabel('std dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
subplot(2,2,4)
surf(stds_par2(:,:,2), 'FaceAlpha',0.5)
title('std dev I')
xlabel('h_E')
ylabel('h_I')
zlabel('st dev')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)


saveas(gcf, 'figures/2Drate_meanstd_mesh_02-1.png')


%% Plot b = (0.55:0.7) vs b = 1

%b-range2 1 is
I_range2(17)

%b-range2 0.55, 0.6, 0.65, 0.7 is idx 8, 9, 10, 11
I_range2(8)

num = [17,8,9,10,11];

figure;
for n=1:length(num)
    
    idx = num(n);

    %E only
    subplot(1,2,1)
    plot(E_range2(:),mean_par2(idx,:,1), 'LineWidth', 1);
    hold on
    legend(strcat('b =', num2str(I_range2(num)')))
    title("mean rate E")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(E_range2(:),stds_par2(idx,:,1), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(I_range2(num)')))
    title("std dev. rate E")
    ylabel("rate")
    xlabel("a-range")
end


figure;
for n=1:length(num)    
    
    idx = num(n);
    
    %I only
    subplot(1,2,1)
    plot(E_range2(:),mean_par2(idx,:,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =',num2str(I_range2(num)')))
    title("mean rate I")
    ylabel("rate")
    xlabel("a-range")

    subplot(1,2,2)
    plot(E_range2(:),stds_par2(idx,:,2), 'LineWidth', 1)
    hold on
    legend(strcat('b =', num2str(I_range2(num)')))
    title("std dev. rate I")
    ylabel("rate")
    xlabel("a-range")
end






