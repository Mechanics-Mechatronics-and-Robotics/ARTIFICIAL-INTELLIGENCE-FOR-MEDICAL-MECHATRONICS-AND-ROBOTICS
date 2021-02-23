%% Task 1: Sine function approximation

% The task is the simple way to study MATLAB Neural Network Start module
% (nnstart) as a part of Deep Learning Toolbox 
%
% The task deals with sin(x) approximation. The main question is how well
% an ANN in data interpoltion and extrapolation. 
%
%  For this task, you will need to change code in this file
%  in the highlighted regions and create a function that represents the
%  trained ANN

%% Initialization and settings
clear ; close all; clc

%Data for training
N=100;%number of inputs and targets
range=[0 2*pi];%range of inputs
NEnh=1000; rangeEnh=[0 4*pi];% enhanced values

%% 1. Dataset
X=linspace(range(1),range(2),N);% inputs
Y=sin(X);%targets
m = length(Y); % number of training examples

%% 2. Visualization
figure
plot(X,Y,'xr');
grid on

%% 3. Made and save the ANN using nnstart module
nnstart% start nnstart
%Train the ANN and save it using "MATLAB Matrix-Only Funtion" 

%Test the ANN in a point
x=pi/3;
y=sin(x)
[h] = myNeuralNetworkFunction(x)
Error=y-h
%% 4. Testing and visualization with enhanced data
X=0;Y=0;H=0;%just in case

% ====================== YOUR CODE HERE ======================
% Instructions: Compute enhanced X and Y matrices using rangeEnh.
%               Compute H using myNeuralNetworkFunction.m 
% Hint: You can use whole matrix X to compute H in one line



% =========================================================================
plotData(X,Y,H)
