function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;

J = sum(power(X*theta-y,2))/2/m;

end
