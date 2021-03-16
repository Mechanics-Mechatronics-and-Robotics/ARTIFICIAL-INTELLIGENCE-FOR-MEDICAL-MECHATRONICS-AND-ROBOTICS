function [lgraph] = unetplus(imsize,numClasses,classWeights,tbl)

lgraph = layerGraph();

tempLayers = [
    imageInputLayer(imsize,"Name","ImageInputLayer")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    dropoutLayer(0.5,"Name","Encoder-Stage-3-DropOut")
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-1")
    convolution2dLayer([3 3],512,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-2")
    dropoutLayer(0.5,"Name","Bridge-DropOut")
    transposedConv2dLayer([2 2],256,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation")
    convolution2dLayer([3 3],256,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],256,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],128,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv2dLayer([2 2],64,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    convolution2dLayer([1 1],2,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    softmaxLayer("Name","Softmax-Layer")
    pixelClassificationLayer("Name","Segmentation-Layer",...
                             'Classes',tbl.Name,...
                                'ClassWeights',classWeights)];
    %pixelClassificationLayer("Name","Segmentation-Layer")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-DropOut");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","Decoder-Stage-1-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","Decoder-Stage-2-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","Decoder-Stage-3-DepthConcatenation/in1");

end

