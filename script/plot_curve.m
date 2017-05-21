maxZ=[0,0,0,0];
for i=0.7:0.001:0.8
    for j=0.5:0.001:0.6
        for k=0.5:0.001:0.6
            x=[i,j,k,1];
            Z=forward_back(x,weight_0,bias_0,weight_1,bias_1);
            if(Z>maxZ(1))
                maxZ(1)=Z;
                maxZ(2:4)=[i,j,k];
            end
        end
    end
end