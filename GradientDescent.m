function [theta, J_history] = GradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    temp = X'*(X*theta-y);
    theta = theta - alpha*temp/m;

    % Save the cost J in every iteration    
    J_history(iter) = ComputeCostFunction(X, y, theta);

end

end
