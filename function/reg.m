function [ a,b,r ] = reg( y,x )
%REG 此处显示有关此函数的摘要
%   此处显示详细说明
a=(mean(x.*y)-mean(x)*mean(y))/(mean(x.*x)-mean(x)^2);
b=mean(y)-a*mean(x);
r=(mean(x.*y)-mean(x)*mean(y))/sqrt((mean(x.*x)-mean(x)^2)*(mean(y.*y)-mean(y)^2));

end

