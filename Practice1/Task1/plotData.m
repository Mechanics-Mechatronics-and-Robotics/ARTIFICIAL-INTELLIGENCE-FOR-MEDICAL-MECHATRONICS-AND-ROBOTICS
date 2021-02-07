function plotData(X,Y)
%PLOTDATA Plots the data points x(i) and y(i) into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. 
%
% Hint: You can use 'r' for red, 'b' for blue, 'k' for black, etc.
%                   'x' for crosses,'-' for line, '.' for points, etc.
%       Also, you can make the markers larger by using
%       plot(..., 'rx', 'MarkerSize', 10);
plot(X,Y,'rx')
xlabel('Abdomen 2 circumference')
ylabel('')%<-enter your code here
grid on
% ============================================================

end
