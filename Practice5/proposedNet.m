function [layers] =...
            proposedNet(imsize,numClasses,FilterSize,NumFilters,...
            dropoutVal,strideVal,maxpoolVal)
%Proposed net
layers = [
    imageInputLayer(imsize,"Name","imageinput")
    convolution2dLayer(FilterSize,NumFilters,"Name","conv_1","Padding",...
                         "same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer(maxpoolVal,"Name","maxpool","Padding","same","Stride",...
    strideVal)
    convolution2dLayer(FilterSize,2*NumFilters,"Name","conv_2",...
    "Padding","same")
    reluLayer("Name","relu_2")
    dropoutLayer(dropoutVal,"Name","dropout")
    fullyConnectedLayer(numClasses,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
end

