% author:              Jantine Broek
% collaborator:     Yashar Ahmadian
% goal:                 recreate E-I 2D model of Hennequin add noise and so simulate data of Kohn&Cohen. 
%                          We focused on analysing how the intrinsic dynamics of the network shaped external noise
%                          to give rise to stimulus dependent patterns of response variability.
% model:              stabilized supralinear network model (which is a reduced rate model)    
%% Solve SSN ODE (without noise term) 
% open ReLU.m and ssn_ode.m function files

% Time vector
dt = 0.00003;
T0 = 0;
Tf = 0.5; % 5 minutes
tspan = [T0:dt:Tf];

%u_0 = [-80; 60]; %-80 for E, 60 for I
u_0 = ones(2,1);           % set initial condition to 1 to calculate trajectory


%use ode45 to numerical solve eq using Runge-Kutta 4th/5th order
[tout, u] = ode45(@ssn_voltage_ode, tspan, u_0);

% like uniform step sizes: interpolate the result
ui = interp1(tout,u,tspan);

x = u(1,:);
y = u(2,:);

figure(1);
%subplot(2,1,1)
plot(tout, u, 'Linewidth', 1.5)
ylabel("voltage - ode45 response")
%legend("E", "I")
%subplot(2,1,2)
%plot(tspan, ui, '-.', 'Linewidth', 1.5)
xlabel("time")
%ylabel("Interpolation response")
title("Time traces of response")
legend("E", "I")


%% SSN with UOP process for every dt
% generate vector lenght u > ode
% add Wiener process

% Membrane time constant 
tau_E = 20/1000; %ms; 20ms for E
tau_I = 10/1000; %ms; 10ms for I
tau = [tau_E; tau_I];

% Input noise std.
sigma_0E = 0.2;     %mV; E cells
sigma_0I = 0.1;     %mV; I cells
sigma_0 = [sigma_0E; sigma_0I]; 

% Initialize output d\eta (W)
W = zeros(2,length(u));

for ii = 1:length(u)-1
    W(:,ii+1) = W(:,ii) + (1./tau).*(-W(:,ii).*dt + (2 * tau .* sigma_0.^2).^0.5.*randn(2,1) * dt.^0.5);
end

figure(2);
plot(tspan, u+transpose(W), 'Linewidth', 1.5)
ylabel("voltage - ode45 response")
xlabel("time")
title("Time traces of response with noise")
legend("E", "I")


%% Look at many starting positions
Tf = 0.25; % 5 minutes
tspan = [T0:dt:Tf];

u0array = repmat([-150:20:100], 2,1);
uarray = zeros(length(u0array),length(tspan), 2);

% figure(3);
% for jj = 1:length(u0array)
%         [tout,u] = ode45(@(t,u) ssn_voltage_ode(t,u), tspan, u0array(:,jj));
%         plot(tspan, u, 'Linewidth', 1.5);
%         hold on
% end
% %uarray = interp2(u); % store each one for plotting later
% 
% xlabel('time (minutes)');
% ylabel('rate');
% %legend("E", "I")
% title('rate over time; different starting points');


for jj = 1:length(u0array)
        [tout,u] = ode45(@(t,u) ssn_voltage_ode(t,u), [T0 Tf], u0array(:,jj));
        uarray(jj,:,:) = interp1(tout,u,tspan); % store each one for plotting later
end

% plot rate over time; different starting points'
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
plot(tspan,uarray(:,:,1), 'Linewidth', 1.5);
xlabel('Time');
ylabel('Voltage');
legend(strcat('start =', num2str(u0array(1,:)')))
title("E")
subplot(1,2,2)
plot(tspan,uarray(:,:,2), 'Linewidth', 1.5);
xlabel('Time');
ylabel('Voltage');
legend(strcat('start =', num2str(u0array(1,:)')))
title("I")

saveas(gcf,'2D_startpos.png')








%% Family of curves for different starting points (DO NOT USE)
y0array = rand(2,10)*200-100; % start them randomly between -5 and 5 in both dimensions
% yarray = zeros(size(y0,2),length(tpositions));

 figure(4);
for ii=1:10
    y0=y0array(:,ii);
    
    [tout,yout] = ode45(@ssn_voltage_ode, tspan, y0);
   
    hold all;
    plot(yout(:,1),yout(:,2));
    
end

xlabel('Voltage E');
ylabel('Voltage I');
title('Voltage over time; different starting points');

%saveas(gcf,'2D_famCurves.png')

% What determines whether a starting point ends up high or low? 
