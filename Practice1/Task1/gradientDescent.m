function [Theta, J_history,Theta_history] = ...
    gradientDescent(X,Y,Theta,alpha,iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   Theta = gradientDescent(X,Y,Theta,alpha,iterations) updates Theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(Y); % number of training examples
param=length(Theta);% number of unknown parameters

J_history = zeros(iterations, 1);
Theta_history=zeros(iterations,param);

gradJ=zeros(param,1);

for iter = 1:iterations

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    
    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, Y, Theta);
    Theta_history(iter,:)=Theta;

end

end
