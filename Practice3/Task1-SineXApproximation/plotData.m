function plotData(X,Y,H)
%PLOTDATA Plots the data points x(i) and y(i) into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels

figure; % open a new figure window

plot(X,Y,'.r',X,H,'-b')
legend('Y','H')
xlabel('Inputs')
ylabel('Targets')%<-enter your code here
grid on
% ============================================================

end
