% author: Jantine Broek
% date: Dec 2017
% Goal: Simulating the Ornstein-Uhlenbeck process
% Reference: https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


%% Simulation
th = 1;
mu = 1.2;
sig = 0.3;
dt = 1e-2;
t = 0:dt:2;                         % Time vector
x = zeros(1,length(t));     % Allocate output vector, set initial condition
rng(1);                             % Set random seed

for i = 1:length(t)-1
    x(i+1) = x(i)+th*(mu-x(i))*dt+sig*sqrt(dt)*randn;
end

figure;
plot(t,x);
hold on 

%% Solution in terms of integral
th = 1;
mu = 1.2;
sig = 0.3;
dt = 1e-2;
t = 0:dt:2;                         % Time vector
x0 = 0;                             % Set initial condition
rng(1);                              % Set random seed
W = zeros(1,length(t));     % Allocate integrated W vector

for i = 1:length(t)-1
    W(i+1) = W(i)+sqrt(dt)*exp(th*t(i))*randn; 
end

ex = exp(-th*t);
x = x0*ex+mu*(1-ex)+sig*ex.*W;

%figure;
plot(t,x);
hold on

%% Analytical solution
% with Wiener increments and ``scaled-time transformed'' Wiener process
th = 1;
mu = 1.2;
sig = 0.3;
dt = 1e-2;
t = 0:dt:2;                             % Time vector
x0 = 0;                                 % Set initial condition
rng(1);                                  % Set random seed
W = zeros(1,length(t));        % Allocate integrated W vector

for i = 1:length(t)-1
    W(i+1) = W(i)+sqrt(exp(2*th*t(i+1))-exp(2*th*t(i)))*randn;
end

ex = exp(-th*t);
x = x0*ex+mu*(1-ex)+sig*ex.*W/sqrt(2*th);

%figure;
plot(t,x);


