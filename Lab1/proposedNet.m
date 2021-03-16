function [layers] = proposedNet(imsize,numClasses,classWeights,tbl,...
            maxPoolDownsample2x,transposedConvUpsample2x)
%Proposed network
layers = [
            imageInputLayer(imsize)
            conv
            reluLayer()
            maxPoolDownsample2x
            conv
            reluLayer()
            maxPoolDownsample2x
            conv
            reluLayer()
            maxPoolDownsample2x
            conv
            reluLayer()
            maxPoolDownsample2x
            
            % Enter your code here and add more layers
            
            
            %%%%%%%%%%%%%%%%%%%%%%%

            transposedConvUpsample2x
            reluLayer()
            transposedConvUpsample2x
            reluLayer()
            transposedConvUpsample2x
            reluLayer()
            transposedConvUpsample2x
            reluLayer()

            convolution2dLayer(1,numClasses);
            softmaxLayer()
            pixelClassificationLayer('Classes',tbl.Name,...
                                'ClassWeights',classWeights)
            ];
end

