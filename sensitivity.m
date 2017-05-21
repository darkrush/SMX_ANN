function [sen]= sensitivity (weight_0,weight_1)
weight_0=abs(weight_0);
weight_1=abs(weight_1);
tmp1=weight_0./repmat(sum(weight_0,1),4,1)*weight_1;
sen=tmp1/norm(tmp1,1);
end