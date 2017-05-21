function [ y,d_weight_0,d_bias_0,d_weight_1,d_bias_1] = forward_back(x,weight_0,bias_0,weight_1,bias_1,y_)
%FORWARD 此处显示有关此函数的摘要
%   此处显示详细说明
batch_size=size(x,1);
middle_data=1./(1+exp(-(x*weight_0+repmat(bias_0,[batch_size,1]))));
y=(middle_data*weight_1+repmat(bias_1,[batch_size,1]));
if nargin==5
    return
end

Pl_Poutput=sign(y-y_)/batch_size;
d_bias_1=sum(Pl_Poutput);
d_weight_1=middle_data'*Pl_Poutput;


Pl_Pmd=Pl_Poutput*weight_1';
Pl_Pmd=(1-middle_data).*middle_data.*Pl_Pmd;
d_bias_0=sum(Pl_Pmd);
d_weight_0=x'*Pl_Pmd;

end

