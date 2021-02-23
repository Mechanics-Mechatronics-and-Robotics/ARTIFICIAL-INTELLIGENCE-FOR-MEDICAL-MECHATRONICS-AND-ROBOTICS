%% Task 2: Binary classification uning an ANN

% Pattern recognition is the process of training a neural network
% to assign the correct target classes to a set of input patterns. 
% Once trained the network can be used to classify patterns it has not seen
% before.  This dataset can be used to design a neural network that
% classifies cancers as either benign or malignant depending on the
% characteristics of sample biopsies. 

% The dataset is taken from the MATLAB built-in datasets 
%
%  Instructions
%  You will need to complete the following functions:
%
%     F_score.m
%
%  For this task, you will need to change code in this file
%  in the highlighted regions and create a function that represents the
%  trained ANN

%% Initialization and settings
clear ; close all; clc

%Settings, including net.trainParam
N=600;%number of tests for the ANN

hiddenLayersSizes = [10 10];%sizes of hidden layers
maxEpochs=100;%maximum number of Epochs
performanceGoal=0;%performance goal
minGrad=1e-6;%minimal value of the gradient 
maxValChecks=1e3;%maxim number of the validation iterations
divideDataSet=[0.7,0.2,0.1];%training,validation and test subsets
iterations = 3e6;% # of iterations
alpha = 1e-4;% learning rate

%% 1. Dataset
load cancer_dataset.mat
% CancerInputs is a 9x699 matrix defining nine attributes of 699 biopsies:
%1. Clump thickness 
%2. Uniformity of cell size 
%3. Uniformity of cell shape 
%4. Marginal Adhesion 
%5. Single epithelial cell size 
%6. Bare nuclei 
%7. Bland chomatin 
%8. Normal nucleoli 
%9. Mitoses  
%cancerTargets - a 2x699 matrix where each column indicates a correct 
%category with a one in either element 1 or element 2.
%1. Benign
%2. Malignant  

%Divide data into 2 subsets
inputs=cancerInputs(:,1:N);%takes N first samples 
targets=cancerTargets(:,1:N);%takes N first samples
inputs_test=cancerInputs(:,N+1:end);%takes the rest samples
targets_test=cancerTargets(:,N+1:end);%takes the rest samples

%% 2. The ANN design and tuning

%Create the ANN
net = patternnet(hiddenLayersSizes);
view(net)

%The ANN settings
net.trainParam.epochs = maxEpochs; 
net.trainParam.goal = performanceGoal; 
net.trainParam.max_fail = maxValChecks; 
net.trainParam.min_grad=minGrad;
net.divideParam.trainRatio = divideDataSet(1);
net.divideParam.valRatio = divideDataSet(2);
net.divideParam.testRatio = divideDataSet(3);

%% 3. The ANN training, validation and testing 
[net,tr] = train(net,inputs,targets);
H=net(inputs);%the ANN predictions
figure
plotperform(tr)
%plotconfusion(H,targets)

%% 4. Test the ANN using data from the inputs_val and targets_val subsets
H=0;
H=net(inputs_test);%the ANN predictions
H=round(H);%round to zero or one

Accuracy=accuracy_calc(net,H,targets_test)
%You should fix the function bellow
F1=F_score(net,H,targets_test)

