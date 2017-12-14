n = -70:0.1:80; %vector
M = randn(1000,2);

a1 = ReLU(M);
a2 = poslin(n);

figure;
subplot(2,1,1)
plot(n,a1)
subplot(2,1,2)
plot(n, a2)

%figure;
% V_rest = -70; 
% b = ReLU(n - V_rest);
% plot(n, b)