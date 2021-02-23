function accuracy=accuracy_calc(net,H,targets)
%Compute accuracy for binary classification

n=length(targets);%# of samples
accuracy=sum(H(1,:)==targets(1,:))/n;

end

