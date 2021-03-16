clear all
clc

%The goal of the lab.task is semantic segmentation of medical images.  
%Want to know more?
% See "Semantic Segmentation With Deep Learning" on MATLAB Help

% The dataset is submitted on GitHub: https://github.com/Mechanics-Mechatronics-and-Robotics/ARTIFICIAL-INTELLIGENCE-FOR-MEDICAL-MECHATRONICS-AND-ROBOTICS/tree/main/Practice5
% The dataset consists of 2 folders: the train and the test ones.
%
% Instructions
% You will need to complete the following script:
%      
%     Main.m
%
% You may enhance the function
%
%     proposedNet.m
%

%% 1.Preprocess: prepare and load labeled data

cd %'G:\VCS_nail_Fold'
saveDir=cd;%save main directory

%Use "Image Labeler" Apps to add labels in images.
%Then export labels to file  
load('gTruth')
gTruth
labelsInfo=gTruth.LabelDefinitions

%% 2. Settings
%Labeling
classNames = [gTruth.LabelDefinitions{1:2,1}]

%Augmentation
ang=[-1 1];
sc=[1 1];
sh=[-1 1];
transl=[-1 1]; 

%Semantic segmentation
%Proposed net
 %Input 
  imsize=[224 224 1];
  numClasses = size(gTruth.LabelDefinitions,1);

 %Downsampling
  numFilters = 8;
  filterSize = 9;
  PaddingSize=4;
  poolSize=2;
  StrideSize=2;
  conv=convolution2dLayer(filterSize,numFilters,'Padding',PaddingSize);
  maxPoolDownsample2x=maxPooling2dLayer(poolSize,'Stride',StrideSize);
 
 %Upsampling
  filterSize = 4;
  StrideSize=2;
  CroppingSize=1;% is set to 1 to make the output size = 2 * input size
  transposedConvUpsample2x=...
    transposedConv2dLayer(filterSize,numFilters,'Stride',StrideSize,...
    'Cropping',CroppingSize);

%Training
trainAlgorithm = 'adam'; %'sgdm', 'adam'
InitLearnRate = 1e-4;
MaxEp = 500;
MiniBatchS = 3;
LearnRateDropFact = 0.9;
LearnRateDropPer = 100;

%U-Net
encoderDepth=3;% if unet

%The rest
numObservations=4;
netType='unetplus';% 'proposed' or 'unetplus'

%% 3. Create training data
augmenter = imageDataAugmenter( ...
    'RandRotation',ang, ...
    'RandScale',sc,...
    'RandXShear',sh,...
    'RandYShear',sh,...
    'RandXTranslation',transl,...
    'RandYTranslation',transl);
trainingData = pixelLabelImageDatastore(gTruth);%,'DataAugmentation',augmenter);

%Analyze training data and correct class weights
tbl = countEachLabel(trainingData)
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency

%% 4. Create the net
switch netType
    case 'Proposed'
        [layers] = proposedNet(imsize,numClasses,classWeights,tbl,...
            maxPoolDownsample2x,transposedConvUpsample2x)
        analyzeNetwork(layers)
    case 'unetplus'
        [lgraph] = unetplus(imsize,numClasses,classWeights,tbl)
        analyzeNetwork(lgraph)
end

%% 5. Train the net
switch netType
    case 'Proposed'
        opts = trainingOptions('sgdm', ...
            'InitialLearnRate',InitLearnRate, ...
            'MaxEpochs',MaxEp, ...
            'MiniBatchSize',MiniBatchS,...
            'LearnRateDropFactor',LearnRateDropFact, ...
            'LearnRateDropPeriod',LearnRateDropPer, ...
            'Plots','training-progress');
        net = trainNetwork(trainingData,layers,opts);
    case 'unetplus'
        opts = trainingOptions(trainAlgorithm, ...
            'InitialLearnRate',InitLearnRate, ...
            'MaxEpochs',MaxEp, ...
            'MiniBatchSize',MiniBatchS,...
            'LearnRateDropFactor',LearnRateDropFact, ...
            'LearnRateDropPeriod',LearnRateDropPer, ...
            'Plots','training-progress');
        net = trainNetwork(trainingData,lgraph,opts)
end

%% 6. Check the trained net using 1 image
testImage = imread('img1_00000_00000000243.png');
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure
imshow(B)

%Create a binary mask of the first class
Mask = C == classNames{1};
figure
imshowpair(testImage, Mask,'montage')

%% 7. Check the trained net using all images in the Test folder

% Enter your code here
%
%     Create datastore of images from the Test folder
%     Segment all the images using the trained net
%     Save segmented images in a new folder
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


