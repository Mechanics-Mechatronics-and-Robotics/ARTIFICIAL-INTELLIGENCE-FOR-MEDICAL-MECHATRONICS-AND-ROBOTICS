% Task 1: Deep convolutional neural networks (CNN) and transfer learning

% The goal of the task is an application of pretrained deep convolutional
% networks to pattern recognition. 
% The dataset is submitten on GitHub: https://github.com/Mechanics-Mechatronics-and-Robotics/ARTIFICIAL-INTELLIGENCE-FOR-MEDICAL-MECHATRONICS-AND-ROBOTICS/tree/main/Practice4
% The dataset consists of 2 folders: the train and the test ones.
% 
% Instructions
% You will need to complete or create the following script and functions:
%     Main.m
%     resnet18.m
%     resnet18pretrained.m
%
%% 0.Settings
clear; close all; clc

%Directories
cd 'E:\work\21_AI4MMR\Practice4'
trainDir='SmilesTrain';
testDir='SmilesTest';

%Enter your code here
%paramDir=''
%%%%%%%%%%%%%%%%%%%%%%

%Additional setting
n=16;%number of images in a preview

%DataSet
imsize=[30 30 3];% images size
imsize2=[224 224 3];% alternative images size
numClasses=2;%number of classes
trainRatio=0.7;%train ratio

%Augmentation
ang=[-5 5];%angle
sc=[1 1];%scale
sh=[-1 1];%shear
transl=[-1 1];%translation

%Train options
MiniBatchS=10;% mini-batch size
MaxE=2;%number of epoches
InitialLearnR=1e-4;%initial learn rate (alpha)
ValidationF=3;%validation frequency

%Choose the net type
netType='resnet18basic';% 'resnet18basic' or 'resnet18pretrained'

%% 1. Create Data Store
%for the training and validation
imds = imageDatastore(trainDir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%for test
imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% 2. Create training and validation datasets. Augment data
%Devide dataset into 2 parts to create training and validation datasets
[imdsTrain,imdsValidation] = ...
    splitEachLabel(imds,trainRatio,'randomized'); 

%Augment dataset
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',ang, ...
    'RandScale',sc,...
    'RandXShear',sh,...
    'RandYShear',sh,...
    'RandXTranslation',transl,...
    'RandYTranslation',transl);
 %according to the netType option
switch netType
    case 'resnet18basic'
        augimdsTrain = augmentedImageDatastore(imsize2,...
            imdsTrain,'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(imsize2,...
            imdsValidation,'DataAugmentation',imageAugmenter);
        augimdsTest = augmentedImageDatastore(imsize2,...
            imdsTest,'DataAugmentation',imageAugmenter);
    case 'resnet18pretrained'
        augimdsTrain = augmentedImageDatastore(imsize2,...
            imdsTrain,'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(imsize2,...
            imdsValidation,'DataAugmentation',imageAugmenter);
        augimdsTest = augmentedImageDatastore(imsize2,...
            imdsTest,'DataAugmentation',imageAugmenter);
end
        
%Visualisation
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,n);
figure
for i = 1:n
    subplot(sqrt(n),sqrt(n),i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% 3. Create the CNN using Deep Network Designer and generate code
%Use deepNetworkDesigner command to start
%Dowload net by typing net=resnet18() in the Command Window 
%Generate code and edit resnet18.m when netType = 'resnet18'
%Generate code with Pretrained Parameters and edit resnet18pretrained.m
%when netType ='resnet18pretrained'

%% 4. Download and train the CNN

%Training options
opts = trainingOptions('sgdm', ...
    'MiniBatchSize',MiniBatchS, ...
    'MaxEpochs',MaxE, ...
    'InitialLearnRate',InitialLearnR, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',ValidationF, ...
    'Verbose',true, ...
    'Plots','training-progress');
%Download and train using the generated code proposedNet.m or alexNet.m         
switch netType
    case 'resnet18basic'
        [lgraph] =...
            resnet18basic(imsize2,numClasses);
        analyzeNetwork(lgraph)
        net = trainNetwork(augimdsTrain,lgraph,opts);
    case 'resnet18pretrained'
        [lgraph] = resnet18pretrained(imsize2,numClasses,paramDir);
        analyzeNetwork(lgraph)
        net = trainNetwork(augimdsTrain,lgraph,opts);
end
%% 5. Test the CNN

[YPred] = classify(net,augimdsTest);%predictions
YTest = imdsTest.Labels;%targets

%Visualisation
idx = randperm(numel(imdsTest.Files),n);
figure
for i = 1:n
    subplot(sqrt(n),sqrt(n),i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = ([YPred(idx(i)),YTest(idx(i))]);
    title(string(label));
end

%Accuracy
accuracy = mean(YPred == YTest)