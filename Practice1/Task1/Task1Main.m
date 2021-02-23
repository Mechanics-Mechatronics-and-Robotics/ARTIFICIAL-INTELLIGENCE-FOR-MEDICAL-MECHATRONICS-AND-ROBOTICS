%% Task 1: Linear Regression

% The task is based on the Machine Learning online course by Andrew Ng 
% https://www.coursera.org/learn/machine-learning
% The dataset is taken from the MATLAB built-in datasets 
%
%  Instructions
%  You will need to complete the following functions:
%
%     plotData.m
%     computeCost.m
%     gradientDescent.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.

%% Initialization and settings
clear ; close all; clc

% Some gradient descent settings
iterations = 3e6;% # of iterations
alpha = 1e-4;% learning rate

%% 1. Dataset
load bodyfat_dataset.mat
% bodyfatInputs is a 13x252 matrix defining 13 attributes for 252 people.  
%1. Age (years) 
%2. Weight (lbs) 
%3. Height (inches) 
%4. Neck circumference (cm) 
%5. Chest circumference (cm) 
%6. Abdomen 2 circumference (cm) 
%7. Hip circumference (cm) 
%8. Thigh circumference (cm) 
%9. Knee circumference (cm) 
%10. Ankle circumference (cm) 
%11. Biceps (extended) circumference (cm) 
%12. Forearm circumference (cm) 
%13. Wrist circumference (cm)
% bodyfatTargets is a 1x252 matrix of associated body fat percentages.
X=bodyfatInputs(6,:);%refers to Abdomen 2 circumference (cm)
Y=bodyfatTargets;
m = length(Y); % number of training examples
%% 2. Visualization
% Plot Data
% Note: please complete the code in plotData.m
plotData(X,Y);

%% 3. Cost function and Gradient descent
X = [ones(m, 1), X']; % add a column of ones to transposed X
Y=Y';% transposed Y
% Initialize fitting parameters for the hypothesis H = X*Theta
Theta = zeros(2, 1); 

fprintf('\nTesting the cost function ...\n')
% Compute and display initial cost
J = computeCost(X,Y,Theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 218\n');

fprintf('\nRunning Gradient Descent ...\n')
% Run gradient descent
[Theta, J_history,Theta_history] = ...
    gradientDescent(X,Y,Theta,alpha,iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', Theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -38.6\n  0.623\n\n');

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*Theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%% 4. Visualizing J(Theta)
fprintf('Visualizing J(Theta) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(min(Theta_history(:,1)),...
                       max(Theta_history(:,1)), 100);
theta1_vals = linspace(min(Theta_history(:,2)),...
                       max(Theta_history(:,2)), 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, Y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 1000
contour(theta0_vals, theta1_vals, J_vals,...
    logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(Theta_history(:,1), Theta_history(:,2),...
    'rx', 'MarkerSize', 10, 'LineWidth', 1);

figure;
plot(J_history)
xlabel('Number of iteration')
ylabel('Cost function')
grid on