maxP=[0.7490,0.5170,0.5090,1];
xx=-1.5:0.1:1.5;
yy=xx;
zz=xx;
[X,Y] = meshgrid(xx,yy);
Z=X;
figure;
axis_=[1,2;2,3;1,3];
labelname=['PH ';' I ';'PDS'];
for fig=1:3
for idx=1:length(xx)
    for idy=1:length(yy)
        x=maxP;
        x(axis_(fig,1))=xx(idx);
        x(axis_(fig,2))=xx(idy);
        Z(idx,idy)=forward_back(x,weight_0,bias_0,weight_1,bias_1);
    end
end
subplot(2,2,fig);
axis equal;
Z=(Z-min(Z(:)))/(max(Z(:))-min(Z(:)));
surf(X,Y,Z);
xlabel(labelname(axis_(fig,1),:));
ylabel(labelname(axis_(fig,2),:));
zlabel('Degradation rate');
end

result=zeros(length(xx),length(yy),length(zz));
XXX=Z(:);
YYY=XXX;
ZZZ=XXX;
id=1;
for idx=1:length(xx)
    for idy=1:length(yy)
        for idz=1:length(zz)
            x=[xx(idx),yy(idy),zz(idz),1];
            XXX(id)=x(1);
            YYY(id)=x(2);
            ZZZ(id)=x(3);
            id=id+1;
            result(idx,idy,idz)=forward_back(x,weight_0,bias_0,weight_1,bias_1);
        end
    end
end
subplot(2,2,4);
result=result(:);
result_size=10*((result-min(result))/(max(result)-min(result))).^80+0.01;
axis equal;
scatter3(XXX,YYY,ZZZ,result_size,[result_size,1-result_size,1-result_size],'filled');
xlabel(labelname(1,:));
ylabel(labelname(2,:));
zlabel(labelname(3,:));