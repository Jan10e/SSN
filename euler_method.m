function [ u, t ] = euler_method( u, t, dt, df )
%Forward Euler method
%  https://www.mathworks.com/matlabcentral/answers/366717-implementing-forward-euler-method

    for i = 1: n
        % Take the Euler step into the temporary variable
        u = u + df(u, t) * dt;
    end


end

