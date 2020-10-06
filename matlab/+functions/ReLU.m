function [ Xnew ] = ReLU( X )
%This is a rectified linear unit (ReLU) activation function on each element
%of the matrix

% The parameter 'alpha' determines the type of ReLU activation: If
% 'alpha'=0, then it is a simple ReLU; 'alpha'>0, then this is a 'leaky'
% ReLU

[m,n] = size(X);
Xnew = X;
alpha = 0; % this determines the type of ReLU

for i = 1:m
    for j = 1:n
        Xnew(i,j) = max(X(i,j), alpha*X(i,j));
    end
end


end

