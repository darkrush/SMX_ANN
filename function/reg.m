function [ a,b,r ] = reg( y,x )
%REG �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
a=(mean(x.*y)-mean(x)*mean(y))/(mean(x.*x)-mean(x)^2);
b=mean(y)-a*mean(x);
r=(mean(x.*y)-mean(x)*mean(y))/sqrt((mean(x.*x)-mean(x)^2)*(mean(y.*y)-mean(y)^2));

end

