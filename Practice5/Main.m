% Task 1: Deep convolutional neural networks (CNN) design

% The goal of the task is a deep network desing to recognise sad and fun
% smiles. 
% The dataset is submitten on GitHub: https://github.com/Mechanics-Mechatronics-and-Robotics/ARTIFICIAL-INTELLIGENCE-FOR-MEDICAL-MECHATRONICS-AND-ROBOTICS/tree/main/Practice5
% The dataset consists of 2 folders: the train and the test ones.
% 
% Instructions
% You will need to complete or create the following script and functions:
%     
%     alexNet.m
%     proposedNet.m
%
%% 0.Settings
clear; close all; clc

%Directories
cd 'F:\work\21_Методы искусственного интеллекта в медицинской робототехнике\2021\Practice5'
trainDir='SmilesTrain';
testDir='SmilesTest';

%DataSet
imsize=[30 30 3];% images size
imsize2=[227 227 3];% alternative image size
numClasses=2;%number of classes
trainRatio=0.7;%train ratio

%Choose the net type
netType="alexnet";% "proposed" or "alexnet"

%Augmentation
ang=[-5 5];%angle
sc=[1 1];%scale
sh=[-1 1];%shear
transl=[-1 1];%translation

MiniBatchS=10;% mini-batch size
MaxE=30;%number of epoches
InitialLearnR=1e-4;%initial learn rate (alpha)
ValidationF=3;%validation frequency

%Train options (when netType == 'proposed')
FilterSize=[3 3];%kernel
NumFilters=32;
dropoutVal=0.5;%dropout probability
strideVal=[2 2];%stride
maxpoolVal=[2 2];%max pool
        
%Additional setting
n=16;%number of images in a preview
haveYouGeneratedNet="no";% have you already generated the net code: y/n

%% 1. Create Data Store
%for training and validation
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
    case "proposed"
        augimdsTrain = augmentedImageDatastore(imsize,...
            imdsTrain,'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(imsize,...
            imdsValidation,'DataAugmentation',imageAugmenter);
        augimdsTest = augmentedImageDatastore(imsize,...
            imdsTest,'DataAugmentation',imageAugmenter);
    case "alexnet"
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
if haveYouGeneratedNet == "no"
    %Use deepNetworkDesigner command to start
    deepNetworkDesigner
    if netType == "alexnet"
        %Download and edit the alexnet when netType = 'alexnet' 
        %Download
        alexnet
        %Then edit it in the Designer and then generate code alexNet.m
        %[layers] = alexNet(numClasses)
    else 
        %Create your own CNN in the Designer when netType = 'proposed'
        %Generate code proposedNet.m after creation
        %[layers] =...
        %    proposedNet(imsize,numClasses,FilterSize,NumFilters,...
        %    dropoutVal,strideVal,maxpoolVal)
    end
end
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
    case "proposed"
        [layers] =...
            proposedNet(imsize,numClasses,FilterSize,NumFilters,...
            dropoutVal,strideVal,maxpoolVal);
        analyzeNetwork(layers)
        net = trainNetwork(augimdsTrain,layers,opts)
    case "alexnet"
        [layers] = alexNet(numClasses);
        analyzeNetwork(layers)
        net = trainNetwork(augimdsTrain,layers,opts)
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